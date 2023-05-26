import os

os.environ['RWKV_JIT_ON'] = '1'
os.environ['RWKV_CUDA_ON'] = '1'
#os.environ['CUDA_LAUNCH_BLOCKING']='1'

USE_FS = False

import multiprocessing
import time

if USE_FS:
    import fluidsynth
import mido
import tokenizers
import torch
from rwkv.model import RWKV
from rwkv.utils import PIPELINE, PIPELINE_ARGS

from midi_util import (VocabConfig, VocabUtils,
                       generate_program_change_messages, str_to_midi_messages, token_to_midi_message)


def play_midi(queue):
    cfg = VocabConfig.from_json("vocab_config_piano.json")

    if USE_FS:
        fs = fluidsynth.Synth()
        fs.start()

    devices = mido.get_output_names()
    if 'Yamaha CVP-208-1 1' in devices:
        device_name = 'Yamaha CVP-208-1 1'
    else:
        device_name = devices[0]

    # get the midi output port
    with mido.open_output(device_name) as midi_port:
        midi_port.reset()
        for i in range(128):
            midi_port.send(mido.Message('note_off', note=i))

        if USE_FS:
            sfid = fs.sfload("C:/Program Files/VideoLan/VLC/GeneralUser GS v1.471.sf2")

        # send program change messages
        program_change_messages = generate_program_change_messages(cfg)
        for msg in program_change_messages:
            #fs.program_select(msg.channel, sfid, msg.program, 0)
            # or
            #fs.cc(msg.channel, 0, msg.program)
            # or
            midi_port.send(msg)

        # now, in the main process, read from the fifo and play the notes
        last_time = time.time()
        active_notes = {}  # {(channel, note): time}
        while True:
            if queue.qsize() > 0:
                msg = queue.get()
                delta = mido.tick2second(msg.time, 480, 500000)
                if delta > 0:
                    time_to_sleep = delta - (time.time() - last_time) if last_time != 0 else delta
                    if time_to_sleep > 0:
                        time.sleep(time_to_sleep)
                    last_time = time.time()

                    # manage midi note state
                    # turn off notes before repeating them
                    if (msg.channel, msg.note) in active_notes:
                        midi_port.send(mido.Message('note_off', channel=msg.channel, note=msg.note))
                        del active_notes[(msg.channel, msg.note)]
                    # keep track of active notes
                    if msg.type == "note_on":
                        active_notes[(msg.channel, msg.note)] = time.time()
                    elif msg.type == "note_off":
                        if (msg.channel, msg.note) in active_notes:
                            del active_notes[(msg.channel, msg.note)]
                    # turn off notes that the model may have forgotten to turn off
                    notes_to_remove = []
                    for (channel, note), t in active_notes.items():
                        if last_time - t > 5.0:
                            midi_port.send(mido.Message('note_off', channel=channel, note=note))
                            notes_to_remove.append((channel, note))
                    for note in notes_to_remove:
                        del active_notes[note]
                    
                    msg.time = 0
                    midi_port.send(msg)
            else:
                time.sleep(0.1)

if __name__ == '__main__':
    cfg = VocabConfig.from_json("vocab_config_piano.json")
    utils = VocabUtils(cfg)

    model = RWKV(model="rwkv-425", strategy='cuda fp16')
    pipeline = PIPELINE(model, "tokenizer-midipiano.json")
    tokenizer: tokenizers.Tokenizer = tokenizers.Tokenizer.from_file("tokenizer-midipiano.json")

    start_token = tokenizer.encode('<start>').ids[0]
    end_token = tokenizer.encode('<end>').ids[0]
    pad_token = tokenizer.encode('<pad>').ids[0]

    queue = multiprocessing.Queue()
    max_queue_size = 256

    # start the thread
    p = multiprocessing.Process(target=play_midi, args=(queue,), daemon=True)
    p.start()

    # chopin nocturne op 9 no 2
    #ctx = "<start> p:46:8 t88 p:27:6 t13 p:4f:b t103 p:37:5 t1 p:3f:7 t71 p:43:7 t1 p:3a:7 t2 p:3f:7 t70 p:33:8 t78 p:4d:b t3 p:38:7 t2 p:3e:8 t61 p:4f:c t4 p:3e:8 t1 p:3b:7 p:44:7 t65 p:4d:b t1 p:27:7 t68 p:37:6 p:3f:8 t60 p:43:7 t2 p:3a:7 p:3f:8 t69 p:4b:b t3 p:27:0 p:33:0 p:37:0 p:38:0 p:3a:0 p:3b:0 p:3e:0 p:3f:0 p:43:0 p:44:0 p:46:0 p:4d:0 p:4f:0 t2 p:26:5 p:4f:0 t2 p:3a:0 t1 p:4d:0 t62 p:37:6 t1 p:3f:8 t76 p:46:a t4 p:43:7 t1 p:3f:7 t1 p:3a:7 t75 p:24:7 t7 p:26:0 p:3a:0 p:3f:0 p:46:0 t2 p:37:0 p:4b:0 t2 p:4f:b t1 p:43:0 t66 p:40:8 t1 p:37:7 t64 p:43:7 t2 p:3a:6 p:40:6 t12 p:48:9 t11 p:49:a t11 p:48:b t11 p:47:b t14 p:48:c t21 p:30:9 t2 p:24:0 p:37:0 p:3a:0 p:40:0 p:43:0 p:47:0 p:48:0 p:49:0 t20 p:54:c t74 p:40:9 t2 p:37:7 t61 p:4f:c t1 p:46:a t1 p:40:9 t3 p:3c:7 t68 p:29:8 p:52:c t7 p:30:0 p:37:0 p:3c:0 p:40:0 p:46:0 p:4f:0 p:54:0 t1 p:48:0 t1 p:40:0 t75 p:35:6 p:3d:7 t82 p:3d:7 p:40:8 t1 p:3a:6 t68 p:50:a t1 p:29:0 p:35:0 p:3a:0 p:3d:0 p:40:0 p:52:0 t1 p:29:6 t67 p:3c:7 t1 p:35:6 t90 p:4f:b t5 p:3c:6 t2 p:41:6 t1 p:38:6 t92 p:4d:b t1 p:2e:8 t7 p:3c:0 t1 p:38:0 p:3c:0 p:41:0 p:4f:0 p:50:0 t3 p:29:0 p:35:0 t62 p:35:6 p:3e:8 p:41:7 t63 p:44:9 t1 p:3e:9 t2 p:3a:8 t16 p:3a:6 t55 p:4f:c t2 p:2f:8 t5 p:2e:0 p:3a:0 p:3e:0 p:41:0 t1 p:35:0 p:3a:0 p:44:0 p:4d:0 t3 p:3e:0 t67 p:37:7 p:41:9 t125 t17"
    #ctx = "<start> p:46:8 t88 p:27:6 t13 p:4f:b t103"
    # suite espanola no 2
    ctx = "<start> p:41:c t94 p:43:d t9 p:41:0 t41 p:44:d t36 p:44:0 t2 p:43:0 t2 p:46:d t31 p:4a:d t14 p:46:0 t14 p:4b:d t41 p:48:c t2 p:4a:0 t6 p:4a:d t7 p:48:0 t1 p:48:d t6 p:4b:0 t99 p:46:d p:48:0 p:4a:0 t125 t10 p:56:d p:5e:d t1 p:52:d t1 p:2e:d p:3a:d t1 p:50:c t125 t45 p:2e:0 p:3a:0 p:46:0 p:50:0 p:5e:0 t2 p:56:0 t1 p:46:c p:52:0 t2 p:3f:8 t4 p:46:0 t2 p:3f:0 t6 p:4b:9 p:4f:8 t13 p:43:8 t2 p:4b:0 t3 p:4f:0 t4 p:43:0 t3 p:48:b t6 p:48:0 t10 p:4b:8 p:4f:7 t5 p:4f:0 t2 p:4b:0 t7 p:43:5 t9 p:46:a t5 p:46:0 t3 p:43:0 t7 p:4b:8 p:4f:6 t4 p:4f:0 t6 p:4b:0 t1 p:43:6 t6 p:43:0 t5 p:4a:b t1 p:46:8 t6 p:4a:0 t2 p:46:0 t4 p:4b:9 p:4f:8 t5 p:4b:0 p:4f:0 t9 p:43:7 t8 p:43:0 t5 p:4b:b t5 p:4b:0 t10 p:4b:8 p:4f:7 t1 p:4b:0 t4 p:4f:0 t7 p:43:7 t5 p:43:0 t6 p:48:a t5 p:48:0 t9 p:4f:8 t1 p:4b:7 t6 p:4b:0 p:4f:0 t6 p:43:7 t5 p:43:0 t5 p:46:b t2 p:3f:8 t5 p:46:0 t1 p:3f:0 t5 p:4b:9 p:4f:9 t7 p:4f:0 t1 p:4b:0 t4 p:43:8 t5 p:43:0 t9 p:48:a t5 p:48:0 t9 p:4b:9 p:4f:8 t6 p:4b:0 p:4f:0 t6 p:43:7 t7 p:43:0 t4 p:46:a t4 p:46:0 t11 p:4b:9 p:4f:7 t6 p:4b:0 t1 p:4f:0 t4 p:43:7 t4 p:43:0 t7 p:46:8 p:4a:b p:4d:9 t1 p:46:0 p:4d:0 t4 p:4a:0 t7 p:4b:9 p:4f:8 t5 p:4f:0 t1 p:4b:0 t7 p:43:7 p:46:5 t5 p:46:0 t3 p:43:0 t5 p:4b:b t13 p:4b:0 t1 p:4f:8 t1 p:4b:7 t4 p:4b:0 t1 p:4f:0 t6 p:43:8 t4 p:43:0 t8 p:48:9 t5 p:48:0 t7 p:4b:9 p:4f:7 t5 p:4b:0 t3 p:4f:0 t7 p:43:5 t6 p:43:0 t3 p:46:a t2 p:41:8 t4 p:46:0 t4 p:41:0 t3 p:4b:9 p:4e:8 t3 p:50:8 t7 p:4b:0 t2 p:44:8 t2 p:4e:0 t10 p:44:0 p:47:a t1 p:50:0 t4 p:47:0 t8 p:4b:a t1 p:50:9 t8 p:50:0 t2 p:44:8 p:4b:0 t7 p:44:0 t7 p:46:b t5 p:46:0 t10 p:4a:a t1 p:50:a t9 p:4a:0 t1 p:44:a p:50:0 t4 p:44:0 t7 p:49:d t5 p:49:0 t8 p:4c:a p:50:a t8 p:4c:0 t5 p:44:a t3 p:50:0 t2 p:44:0 t7 p:46:c t6 p:46:0 t5 p:4c:9 t5 p:4c:0 t1 p:50:8 t10 p:50:0 t1 p:44:7 t5 p:44:0 t7 p:4a:c t7 p:4a:0 t5 p:4d:a p:50:a t7 p:4d:0 t2 p:50:0 t2 p:46:a t4 p:46:0 t10 p:4c:d t11 p:4c:0 t3 p:4f:b t1 p:52:a t5 p:4f:0 t7 p:49:a t2 p:52:0 t2 p:49:0 t7 p:46:b t7 p:46:0 t6 p:4f:a t1 p:52:a t11 p:49:a t1 p:4f:0 t1 p:52:0 t3 p:49:0 t6 p:4d:b t7 p:4d:0 t6 p:52:a t1 p:50:9 t5 p:52:0 t4 p:4a:a p:50:0 t7 p:4a:0 t4 p:50:d t9 p:50:0 t5 p:4d:a p:54:a t8 p:4d:0 t2 p:4a:a t5 p:4a:0 t3 p:54:0 t6 p:46:b t7 p:46:0 t2 p:53:b t1 p:4d:9 t7 p:4d:0 t2 p:53:0 t4 p:4a:a t4 p:4a:0 t9 p:50:c t5 p:50:0 t9 p:4d:a t1 p:52:a t5 p:4d:0 t1 p:52:0 t6 p:4a:9 t4 p:4a:0 t16 p:46:d t1 p:3f:b t10 p:46:0 t5 p:4b:a t1 p:4f:a t11 p:43:9 t13 p:3f:0 p:43:0 t1 p:48:b p:4f:0 t1 p:46:9 t3 p:4b:0 t1 p:48:0 t9 p:4b:b p:4f:a t7 p:4b:0 p:4f:0 t2 p:46:0 t3 p:43:9 t5 p:43:0 t7 p:46:c t5 p:46:0 t10 p:4b:a p:4f:9 t6 p:4f:0 t4 p:4b:0 t3 p:43:8 t4 p:43:0 t7 p:46:9 p:4a:c t1 p:46:0 t5 p:4a:0 t9 p:4f:a t1 p:4b:a t5 p:4b:0 p:4f:0 t7 p:43:9 p:46:7 t4 p:43:0 p:46:0 t8 p:4b:c t5 p:4b:0 t9 p:4b:9 p:4f:9 t5 p:4f:0 t1 p:4b:0 t7 p:43:8 t4 p:43:0 t6 p:48:a t5 p:48:0 t9 p:4b:9 t1 p:4f:6 t6 p:4b:0 t4 p:43:9 p:4f:0 t6 p:43:0 t7 p:46:d t2 p:3f:a t8 p:3f:0 t1 p:46:0 t3 p:4b:a t2 p:4f:9 t10 p:43:9 t6 p:43:0 p:4b:0 p:4f:0 t8 p:48:b t1 p:46:9 t1 p:46:0 t5 p:48:0 t7 p:4b:a p:4f:9 t6 p:4b:0 t1 p:4f:0 t6 p:43:8 t4 p:43:0 t9 p:46:b t5 p:46:0 t7 p:4b:9 t1 p:4f:8 t5 p:4b:0 p:4f:0 t6 p:43:8 t4 p:43:0 t10 p:46:8 p:4a:b t6 p:4a:0 t3 p:46:0 t3 p:4b:9 p:4f:9 t5 p:4b:0 t1 p:4f:0 t5 p:43:9 p:46:7 t4 p:43:0 t3 p:46:0 t5 p:4b:c t10 p:4b:0 t2 p:4b:8 p:4f:7 t14 p:43:9 t3 p:43:0 t1 p:4b:0 t1 p:4f:0 t7 p:48:a t5 p:48:0 t6 p:4b:9 t1 p:4f:8 t5 p:4b:0 p:4f:0 t6 p:43:9 t4 p:43:0 t7 p:46:c t3 p:41:9 t4 p:41:0 t6 p:4b:a t3 p:50:9 t10 p:44:a t8 p:44:0 p:46:0 p:4b:0 t7 p:47:b p:48:9 p:50:0 t7 p:47:0 t2 p:48:0 t2 p:4b:b t1 p:50:a t8 p:4b:0 p:50:0 t3 p:44:a t5 p:44:0 t7 p:46:c t5 p:46:0 t8 p:4b:9 t1 p:4a:a t1 p:4b:0 p:50:a t7 p:4a:0 t3 p:50:0 t1 p:44:a t5 p:44:0 t7 p:49:d t4 p:49:0 t7 p:4c:a t4 p:50:9 t5 p:4c:0 t5 p:44:a t6 p:44:0 t7 p:46:c t4 p:50:0 t2 p:46:0 t6 p:4c:9 t2 p:50:a t3 p:4c:0 t9"
    # reverie
    #ctx = "<start> p:3a:5 t72 p:3c:5 t46 p:3e:7 t41 p:43:9 t67 p:3e:8 t38 p:3c:7 t42 p:3a:7 t20 p:3a:0 p:3c:0 p:3e:0 p:3e:0 t50 p:3c:7 t43 p:3e:7 t48 p:43:8 t81 p:3e:6 t47 p:3c:8 t61 p:3a:6 t13 p:3a:0 p:3c:0 p:3e:0 p:43:0 t65 p:4f:a t3 p:1f:5 t58 p:3c:7 t36 p:3e:7 t34 p:43:8 t35 p:4a:9 t32 p:3e:7 t39 p:3c:8 t50 p:3a:7 t19 p:1f:0 p:3a:0 p:3c:0 p:3e:0 p:43:0 t62 p:3c:7 t37 p:4c:9 t3 p:3e:6 t30 p:4d:a t6 p:43:8 t33 p:4f:b t28 p:3e:8 t31 p:4c:b t3 p:3c:8 p:43:6 t30 p:4a:b t3 p:3a:8 t39 p:3a:0 p:3c:0 p:3e:0 p:43:0 p:4a:0 p:4c:0 p:4f:0 t1 p:4c:0 t2 p:4c:b t27 p:3c:8 t22 p:48:a t24 p:3e:7 t22 p:4c:a t27 p:43:7 t32 p:4a:9 t33 p:3e:7 t35 p:3c:7 t47 p:3a:7 t19 p:3a:0 p:3c:0 p:3e:0 p:43:0 p:48:0 p:4a:0 p:4c:0 t2 p:3e:0 t3 p:4d:0 t9 p:4f:0 t51 p:3c:6 t38 p:3e:6 t36 p:43:7 t33 p:46:a t28 p:3e:8 t30 p:43:7 p:4a:b t6 p:3c:7 t25 p:3a:9 t28 p:4c:b t6 p:39:9 t1 p:3e:5 t7 p:3a:0 p:3c:0 p:3e:0 p:43:0 p:46:0 p:4a:0 p:4c:0 t16 p:3e:0 t4 p:3a:a t29 p:4d:c t2 p:3c:9 t29 p:41:8 t1 p:4c:0 t37 p:48:a t47 p:3c:7 t39 p:3a:7 t34 p:39:7 t34 p:37:7 t11 p:39:0 p:39:0 p:3a:0 p:3a:0 p:3c:0 p:41:0 t4 p:3c:0 p:4d:0 t15 p:39:8 t31 p:3a:9 t9 p:39:0 t22 p:40:8 t45 p:43:a t38 p:3a:8 t39 p:39:8 t38 p:37:8 t62 p:45:a t5 p:29:7 p:41:4 t15 p:37:0 p:39:0 p:3a:0 p:40:0 p:43:0 p:48:0 t33 p:30:6 t31 p:39:7 t31 p:41:7 t29 p:3e:8 t29 p:39:8 t33 p:3c:8 t1 p:30:5 t31 p:41:7 t32 p:29:8 t34 p:30:7 t34 p:39:7 t32 p:41:7 t35 p:3e:7 t36 p:39:8 t43 p:3c:6 t42 p:41:6 t93 p:51:b t7 p:26:7 t11 p:29:0 p:30:0 p:39:0 p:3c:0 p:3e:0 p:41:0 t27 p:2d:8 t33 p:32:8 t27 p:35:8 t31 p:4c:a t2 p:39:9 t27 p:3c:9 t30 p:40:9 t31 p:3c:9 t31 p:39:9 t34 p:35:8 t39 p:48:a t2 p:39:7 t29 p:4c:a t2 p:3c:9 t36 p:4a:a t2 p:2b:a t8 p:26:0 p:35:0 p:39:0 p:3c:0 p:45:0 p:4c:0 t16 p:32:0 t2 p:32:a t29 p:37:9 t3 p:46:a t27 p:3a:a t1 p:43:a t48 p:2d:0 p:32:0 p:35:0 p:37:0 p:39:0 p:3a:0 p:40:0 p:43:0 p:46:0 p:4a:0 p:4c:0 p:51:0 t1 p:48:0 t1 p:2b:0 t1 p:3c:0"

    state = None
    tokens = tokenizer.encode(ctx).ids
    
    midi_state = None
    token_history = tokens.copy()
    def callback(note: str):
        global midi_state
        msg, midi_state = token_to_midi_message(utils, note, midi_state, end_token_pause=3.0)
        if msg is not None:
            queue.put(msg)
        print(note, end=' ')

    # pump context playback
    for id in tokens:
        note = tokenizer.decode([id])
        callback(note)

    temperature = 1.0
    top_p = 0.8
    top_k = 0
    repetition_penalty = 1.05  # standard LLM repetition penalty. 1.0 = no penalty
    repetition_view_length = 256  # how far back to look for repetitions
    max_penalty = 2.0  # maximum penalty to apply. 1.0 = no penalty
    decay_factor = 0.985  # how much to decay the penalty by, depending on how far back. 1.0 = no decay
    initial_state_weighting = 0.001  # super experimental. I think this biases long generation towards initial context.

    # fast forward through context
    ff_tokens = tokens[:-1] if len(tokens) > 1 else []
    while len(ff_tokens) > 1:
        scores, state = model.forward(ff_tokens[:256], state)
        ff_tokens = ff_tokens[256:]
    tokens = tokens[-1:]
    initial_state = state.copy()

    while True:
        if queue.qsize() < max_queue_size:
            scores, state = model.forward(tokens, state)
            if initial_state_weighting > 0:
                state = [initial_state_weighting * b + (1 - initial_state_weighting) * a for a, b in zip(state, initial_state)]

            scores[pad_token] = -float('inf')
            
            if repetition_penalty != 1.0:
                repetition_context = torch.tensor(token_history[-repetition_view_length:]).to(scores.device)
                # Repetition penalty
                # score = torch.gather(out, 0, repetition_context)
                # score = torch.where(score < 0, score * repetition_penalty, score / repetition_penalty)
                # out.scatter_(0, repetition_context, score)

                # Exponential repetition penalty (accounts for recency and frequency)
                decays = torch.pow(torch.full_like(repetition_context, decay_factor, dtype=scores.dtype), torch.arange(0, repetition_context.shape[-1], device=scores.device, dtype=scores.dtype)).flip(-1)
                mask = torch.zeros_like(scores)
                mask.scatter_add_(-1, repetition_context, decays)
                penalty_factor = torch.pow(torch.where(scores < 0, repetition_penalty, 1 / repetition_penalty), mask)
                penalty_factor = torch.clamp(penalty_factor, torch.tensor(1.0 / max_penalty, device=penalty_factor.device), torch.tensor(max_penalty, device=penalty_factor.device))
                scores = scores * penalty_factor
            
            # sample
            token = pipeline.sample_logits(scores, temperature=temperature, top_p=top_p, top_k=top_k)
            tokens = [token]
            token_history.append(token)
            if len(token_history) > 1024*10:
                token_history = token_history[-1024*8:]
            
            # send note to fluidsynth
            note = tokenizer.decode(tokens, skip_special_tokens=False)
            callback(note)

            # end of sequence
            if token == end_token:
                tokens = [start_token]
                state = None
                token_history = tokens.copy()
                print()
        else:
            time.sleep(0.1)
