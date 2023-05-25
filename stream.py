import os

os.environ['RWKV_JIT_ON'] = '1'
os.environ['RWKV_CUDA_ON'] = '1'
#os.environ['CUDA_LAUNCH_BLOCKING']='1'

import multiprocessing
import time

import fluidsynth
import mido
import tokenizers
import torch
from rwkv.model import RWKV
from rwkv.utils import PIPELINE, PIPELINE_ARGS

from midi_util import (VocabConfig, VocabUtils,
                       generate_program_change_messages, str_to_midi_messages, token_to_midi_message)

def play_midi(queue):
    cfg = VocabConfig.from_json("//wsl.localhost/Ubuntu/home/brian/MIDI-LLM-tokenizer/vocab_config_piano.json")

    fs = fluidsynth.Synth()
    fs.start()

    midi_output = mido.open_output()

    sfid = fs.sfload("C:/Program Files/VideoLan/VLC/GeneralUser GS v1.471.sf2")

    # send program change messages
    program_change_messages = generate_program_change_messages(cfg)
    for msg in program_change_messages:
        fs.program_select(msg.channel, sfid, msg.program, 0)
        # or
        #fs.cc(msg.channel, 0, msg.program)
        # or
        #midi_output.send(msg)

    # now, in the main process, read from the fifo and play the notes
    last_time = time.time()
    while True:
        if queue.qsize() > 0:
            msg = queue.get()
            delta = mido.tick2second(msg.time, 480, 500000)
            if delta > 0:
                time_to_sleep = delta - (time.time() - last_time)
                if time_to_sleep > 0:
                    time.sleep(time_to_sleep)
                last_time = time.time()
                midi_output.send(msg)
        else:
            time.sleep(0.1)

if __name__ == '__main__':
    cfg = VocabConfig.from_json("//wsl.localhost/Ubuntu/home/brian/MIDI-LLM-tokenizer/vocab_config_piano.json")
    utils = VocabUtils(cfg)

    model = RWKV(model="//wsl.localhost/Ubuntu/home/brian/RWKV-LM/RWKV-v4neo/gmpaugP3/rwkv-315", strategy='cuda fp16')
    pipeline = PIPELINE(model, "//wsl.localhost/Ubuntu/home/brian/MIDI-LLM-tokenizer/tokenizer-midipiano.json")
    tokenizer: tokenizers.Tokenizer = tokenizers.Tokenizer.from_file("//wsl.localhost/Ubuntu/home/brian/MIDI-LLM-tokenizer/tokenizer-midipiano.json")

    start_token = tokenizer.encode('<start>').ids[0]
    end_token = tokenizer.encode('<end>').ids[0]
    pad_token = tokenizer.encode('<pad>').ids[0]

    queue = multiprocessing.Queue()
    max_queue_size = 100

    # start the thread
    p = multiprocessing.Process(target=play_midi, args=(queue,), daemon=True)
    p.start()

    ctx = "<start> p:46:8 t88 p:27:6 t13 p:4f:b t103 p:37:5 t1 p:3f:7 t71 p:43:7 t1 p:3a:7 t2 p:3f:7 t70 p:33:8 t78 p:4d:b t3 p:38:7 t2 p:3e:8 t61 p:4f:c t4 p:3e:8 t1 p:3b:7 p:44:7 t65 p:4d:b t1 p:27:7 t68 p:37:6 p:3f:8 t60 p:43:7 t2 p:3a:7 p:3f:8 t69 p:4b:b t3 p:27:0 p:33:0 p:37:0 p:38:0 p:3a:0 p:3b:0 p:3e:0 p:3f:0 p:43:0 p:44:0 p:46:0 p:4d:0 p:4f:0 t2 p:26:5 p:4f:0 t2 p:3a:0 t1 p:4d:0 t62 p:37:6 t1 p:3f:8 t76 p:46:a t4 p:43:7 t1 p:3f:7 t1 p:3a:7 t75 p:24:7 t7 p:26:0 p:3a:0 p:3f:0 p:46:0 t2 p:37:0 p:4b:0 t2 p:4f:b t1 p:43:0 t66 p:40:8 t1 p:37:7 t64 p:43:7 t2 p:3a:6 p:40:6 t12 p:48:9 t11 p:49:a t11 p:48:b t11 p:47:b t14 p:48:c t21 p:30:9 t2 p:24:0 p:37:0 p:3a:0 p:40:0 p:43:0 p:47:0 p:48:0 p:49:0 t20 p:54:c t74 p:40:9 t2 p:37:7 t61 p:4f:c t1 p:46:a t1 p:40:9 t3 p:3c:7 t68 p:29:8 p:52:c t7 p:30:0 p:37:0 p:3c:0 p:40:0 p:46:0 p:4f:0 p:54:0 t1 p:48:0 t1 p:40:0 t75 p:35:6 p:3d:7 t82 p:3d:7 p:40:8 t1 p:3a:6 t68 p:50:a t1 p:29:0 p:35:0 p:3a:0 p:3d:0 p:40:0 p:52:0 t1 p:29:6 t67 p:3c:7 t1 p:35:6 t90 p:4f:b t5 p:3c:6 t2 p:41:6 t1 p:38:6 t92 p:4d:b t1 p:2e:8 t7 p:3c:0 t1 p:38:0 p:3c:0 p:41:0 p:4f:0 p:50:0 t3 p:29:0 p:35:0 t62 p:35:6 p:3e:8 p:41:7 t63 p:44:9 t1 p:3e:9 t2 p:3a:8 t16 p:3a:6 t55 p:4f:c t2 p:2f:8 t5 p:2e:0 p:3a:0 p:3e:0 p:41:0 t1 p:35:0 p:3a:0 p:44:0 p:4d:0 t3 p:3e:0 t67 p:37:7 p:41:9 t125 t17"
    #ctx = "<start> p:46:8 t88 p:27:6 t13 p:4f:b t103"

    state = None
    tokens = tokenizer.encode(ctx).ids
    
    midi_state = None
    token_history = tokens.copy()
    def callback(note: str):
        global midi_state
        msg, midi_state = token_to_midi_message(utils, note, midi_state)
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
    repetition_penalty = 1.05  # standard LLM repetition penalty
    repetition_view_length = 256  # how far back to look for repetitions
    max_penalty = 1.5  # maximum penalty to apply
    decay_factor = 0.99  # how much to decay the penalty by depending on how far back
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
            state = [initial_state_weighting * b + (1 - initial_state_weighting) * a for a, b in zip(state, initial_state)]

            scores[pad_token] = -float('inf')
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
            
            token = pipeline.sample_logits(scores, temperature=temperature, top_p=top_p, top_k=top_k)
            tokens = [token]
            token_history.append(token)
            if len(token_history) > 1024*10:
                token_history = token_history[-1024*8:]
            
            # send note to fluidsynth
            note = tokenizer.decode(tokens)
            callback(note)
        else:
            time.sleep(0.1)
