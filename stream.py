import os

os.environ['RWKV_JIT_ON'] = '1'
os.environ['RWKV_CUDA_ON'] = '1'
#os.environ['CUDA_LAUNCH_BLOCKING']='1'

USE_FS = False

import multiprocessing
import time
import datetime

if USE_FS:
    import fluidsynth
import math
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
        midi_port: mido.ports.BaseOutput
        print("Opened midi port: ", midi_port.name)
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
        while True:
            if queue.qsize() > 0:
                msg = queue.get()
                delta = mido.tick2second(msg.time, 480, 500000)
                if delta > 0:
                    time_to_sleep = delta - (time.time() - last_time) if last_time != 0 else delta
                    if time_to_sleep > 0:
                        time.sleep(time_to_sleep)
                    last_time = time.time()
                msg.time = 0
                midi_port.send(msg)
            else:
                time.sleep(0.1)

if __name__ == '__main__':
    cfg = VocabConfig.from_json("vocab_config_piano.json")
    utils = VocabUtils(cfg)

    model_name = "rwkv-650"
    model_path = f"{model_name}"
    model = RWKV(model=model_path, strategy='cuda fp16')
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
    #ctx = "<start> p:41:c t94 p:43:d t9 p:41:0 t41 p:44:d t36 p:44:0 t2 p:43:0 t2 p:46:d t31 p:4a:d t14 p:46:0 t14 p:4b:d t41 p:48:c t2 p:4a:0 t6 p:4a:d t7 p:48:0 t1 p:48:d t6 p:4b:0 t99 p:46:d p:48:0 p:4a:0 t125 t10 p:56:d p:5e:d t1 p:52:d t1 p:2e:d p:3a:d t1 p:50:c t125 t45 p:2e:0 p:3a:0 p:46:0 p:50:0 p:5e:0 t2 p:56:0 t1 p:46:c p:52:0 t2 p:3f:8 t4 p:46:0 t2 p:3f:0 t6 p:4b:9 p:4f:8 t13 p:43:8 t2 p:4b:0 t3 p:4f:0 t4 p:43:0 t3 p:48:b t6 p:48:0 t10 p:4b:8 p:4f:7 t5 p:4f:0 t2 p:4b:0 t7 p:43:5 t9 p:46:a t5 p:46:0 t3 p:43:0 t7 p:4b:8 p:4f:6 t4 p:4f:0 t6 p:4b:0 t1 p:43:6 t6 p:43:0 t5 p:4a:b t1 p:46:8 t6 p:4a:0 t2 p:46:0 t4 p:4b:9 p:4f:8 t5 p:4b:0 p:4f:0 t9 p:43:7 t8 p:43:0 t5 p:4b:b t5 p:4b:0 t10 p:4b:8 p:4f:7 t1 p:4b:0 t4 p:4f:0 t7 p:43:7 t5 p:43:0 t6 p:48:a t5 p:48:0 t9 p:4f:8 t1 p:4b:7 t6 p:4b:0 p:4f:0 t6 p:43:7 t5 p:43:0 t5 p:46:b t2 p:3f:8 t5 p:46:0 t1 p:3f:0 t5 p:4b:9 p:4f:9 t7 p:4f:0 t1 p:4b:0 t4 p:43:8 t5 p:43:0 t9 p:48:a t5 p:48:0 t9 p:4b:9 p:4f:8 t6 p:4b:0 p:4f:0 t6 p:43:7 t7 p:43:0 t4 p:46:a t4 p:46:0 t11 p:4b:9 p:4f:7 t6 p:4b:0 t1 p:4f:0 t4 p:43:7 t4 p:43:0 t7 p:46:8 p:4a:b p:4d:9 t1 p:46:0 p:4d:0 t4 p:4a:0 t7 p:4b:9 p:4f:8 t5 p:4f:0 t1 p:4b:0 t7 p:43:7 p:46:5 t5 p:46:0 t3 p:43:0 t5 p:4b:b t13 p:4b:0 t1 p:4f:8 t1 p:4b:7 t4 p:4b:0 t1 p:4f:0 t6 p:43:8 t4 p:43:0 t8 p:48:9 t5 p:48:0 t7 p:4b:9 p:4f:7 t5 p:4b:0 t3 p:4f:0 t7 p:43:5 t6 p:43:0 t3 p:46:a t2 p:41:8 t4 p:46:0 t4 p:41:0 t3 p:4b:9 p:4e:8 t3 p:50:8 t7 p:4b:0 t2 p:44:8 t2 p:4e:0 t10 p:44:0 p:47:a t1 p:50:0 t4 p:47:0 t8 p:4b:a t1 p:50:9 t8 p:50:0 t2 p:44:8 p:4b:0 t7 p:44:0 t7 p:46:b t5 p:46:0 t10 p:4a:a t1 p:50:a t9 p:4a:0 t1 p:44:a p:50:0 t4 p:44:0 t7 p:49:d t5 p:49:0 t8 p:4c:a p:50:a t8 p:4c:0 t5 p:44:a t3 p:50:0 t2 p:44:0 t7 p:46:c t6 p:46:0 t5 p:4c:9 t5 p:4c:0 t1 p:50:8 t10 p:50:0 t1 p:44:7 t5 p:44:0 t7 p:4a:c t7 p:4a:0 t5 p:4d:a p:50:a t7 p:4d:0 t2 p:50:0 t2 p:46:a t4 p:46:0 t10 p:4c:d t11 p:4c:0 t3 p:4f:b t1 p:52:a t5 p:4f:0 t7 p:49:a t2 p:52:0 t2 p:49:0 t7 p:46:b t7 p:46:0 t6 p:4f:a t1 p:52:a t11 p:49:a t1 p:4f:0 t1 p:52:0 t3 p:49:0 t6 p:4d:b t7 p:4d:0 t6 p:52:a t1 p:50:9 t5 p:52:0 t4 p:4a:a p:50:0 t7 p:4a:0 t4 p:50:d t9 p:50:0 t5 p:4d:a p:54:a t8 p:4d:0 t2 p:4a:a t5 p:4a:0 t3 p:54:0 t6 p:46:b t7 p:46:0 t2 p:53:b t1 p:4d:9 t7 p:4d:0 t2 p:53:0 t4 p:4a:a t4 p:4a:0 t9 p:50:c t5 p:50:0 t9 p:4d:a t1 p:52:a t5 p:4d:0 t1 p:52:0 t6 p:4a:9 t4 p:4a:0 t16 p:46:d t1 p:3f:b t10 p:46:0 t5 p:4b:a t1 p:4f:a t11 p:43:9 t13 p:3f:0 p:43:0 t1 p:48:b p:4f:0 t1 p:46:9 t3 p:4b:0 t1 p:48:0 t9 p:4b:b p:4f:a t7 p:4b:0 p:4f:0 t2 p:46:0 t3 p:43:9 t5 p:43:0 t7 p:46:c t5 p:46:0 t10 p:4b:a p:4f:9 t6 p:4f:0 t4 p:4b:0 t3 p:43:8 t4 p:43:0 t7 p:46:9 p:4a:c t1 p:46:0 t5 p:4a:0 t9 p:4f:a t1 p:4b:a t5 p:4b:0 p:4f:0 t7 p:43:9 p:46:7 t4 p:43:0 p:46:0 t8 p:4b:c t5 p:4b:0 t9 p:4b:9 p:4f:9 t5 p:4f:0 t1 p:4b:0 t7 p:43:8 t4 p:43:0 t6 p:48:a t5 p:48:0 t9 p:4b:9 t1 p:4f:6 t6 p:4b:0 t4 p:43:9 p:4f:0 t6 p:43:0 t7 p:46:d t2 p:3f:a t8 p:3f:0 t1 p:46:0 t3 p:4b:a t2 p:4f:9 t10 p:43:9 t6 p:43:0 p:4b:0 p:4f:0 t8 p:48:b t1 p:46:9 t1 p:46:0 t5 p:48:0 t7 p:4b:a p:4f:9 t6 p:4b:0 t1 p:4f:0 t6 p:43:8 t4 p:43:0 t9 p:46:b t5 p:46:0 t7 p:4b:9 t1 p:4f:8 t5 p:4b:0 p:4f:0 t6 p:43:8 t4 p:43:0 t10 p:46:8 p:4a:b t6 p:4a:0 t3 p:46:0 t3 p:4b:9 p:4f:9 t5 p:4b:0 t1 p:4f:0 t5 p:43:9 p:46:7 t4 p:43:0 t3 p:46:0 t5 p:4b:c t10 p:4b:0 t2 p:4b:8 p:4f:7 t14 p:43:9 t3 p:43:0 t1 p:4b:0 t1 p:4f:0 t7 p:48:a t5 p:48:0 t6 p:4b:9 t1 p:4f:8 t5 p:4b:0 p:4f:0 t6 p:43:9 t4 p:43:0 t7 p:46:c t3 p:41:9 t4 p:41:0 t6 p:4b:a t3 p:50:9 t10 p:44:a t8 p:44:0 p:46:0 p:4b:0 t7 p:47:b p:48:9 p:50:0 t7 p:47:0 t2 p:48:0 t2 p:4b:b t1 p:50:a t8 p:4b:0 p:50:0 t3 p:44:a t5 p:44:0 t7 p:46:c t5 p:46:0 t8 p:4b:9 t1 p:4a:a t1 p:4b:0 p:50:a t7 p:4a:0 t3 p:50:0 t1 p:44:a t5 p:44:0 t7 p:49:d t4 p:49:0 t7 p:4c:a t4 p:50:9 t5 p:4c:0 t5 p:44:a t6 p:44:0 t7 p:46:c t4 p:50:0 t2 p:46:0 t6 p:4c:9 t2 p:50:a t3 p:4c:0 t9"
    # reverie
    #ctx = "<start> p:3a:5 t72 p:3c:5 t46 p:3e:7 t41 p:43:9 t67 p:3e:8 t38 p:3c:7 t42 p:3a:7 t20 p:3a:0 p:3c:0 p:3e:0 p:3e:0 t50 p:3c:7 t43 p:3e:7 t48 p:43:8 t81 p:3e:6 t47 p:3c:8 t61 p:3a:6 t13 p:3a:0 p:3c:0 p:3e:0 p:43:0 t65 p:4f:a t3 p:1f:5 t58 p:3c:7 t36 p:3e:7 t34 p:43:8 t35 p:4a:9 t32 p:3e:7 t39 p:3c:8 t50 p:3a:7 t19 p:1f:0 p:3a:0 p:3c:0 p:3e:0 p:43:0 t62 p:3c:7 t37 p:4c:9 t3 p:3e:6 t30 p:4d:a t6 p:43:8 t33 p:4f:b t28 p:3e:8 t31 p:4c:b t3 p:3c:8 p:43:6 t30 p:4a:b t3 p:3a:8 t39 p:3a:0 p:3c:0 p:3e:0 p:43:0 p:4a:0 p:4c:0 p:4f:0 t1 p:4c:0 t2 p:4c:b t27 p:3c:8 t22 p:48:a t24 p:3e:7 t22 p:4c:a t27 p:43:7 t32 p:4a:9 t33 p:3e:7 t35 p:3c:7 t47 p:3a:7 t19 p:3a:0 p:3c:0 p:3e:0 p:43:0 p:48:0 p:4a:0 p:4c:0 t2 p:3e:0 t3 p:4d:0 t9 p:4f:0 t51 p:3c:6 t38 p:3e:6 t36 p:43:7 t33 p:46:a t28 p:3e:8 t30 p:43:7 p:4a:b t6 p:3c:7 t25 p:3a:9 t28 p:4c:b t6 p:39:9 t1 p:3e:5 t7 p:3a:0 p:3c:0 p:3e:0 p:43:0 p:46:0 p:4a:0 p:4c:0 t16 p:3e:0 t4 p:3a:a t29 p:4d:c t2 p:3c:9 t29 p:41:8 t1 p:4c:0 t37 p:48:a t47 p:3c:7 t39 p:3a:7 t34 p:39:7 t34 p:37:7 t11 p:39:0 p:39:0 p:3a:0 p:3a:0 p:3c:0 p:41:0 t4 p:3c:0 p:4d:0 t15 p:39:8 t31 p:3a:9 t9 p:39:0 t22 p:40:8 t45 p:43:a t38 p:3a:8 t39 p:39:8 t38 p:37:8 t62 p:45:a t5 p:29:7 p:41:4 t15 p:37:0 p:39:0 p:3a:0 p:40:0 p:43:0 p:48:0 t33 p:30:6 t31 p:39:7 t31 p:41:7 t29 p:3e:8 t29 p:39:8 t33 p:3c:8 t1 p:30:5 t31 p:41:7 t32 p:29:8 t34 p:30:7 t34 p:39:7 t32 p:41:7 t35 p:3e:7 t36 p:39:8 t43 p:3c:6 t42 p:41:6 t93 p:51:b t7 p:26:7 t11 p:29:0 p:30:0 p:39:0 p:3c:0 p:3e:0 p:41:0 t27 p:2d:8 t33 p:32:8 t27 p:35:8 t31 p:4c:a t2 p:39:9 t27 p:3c:9 t30 p:40:9 t31 p:3c:9 t31 p:39:9 t34 p:35:8 t39 p:48:a t2 p:39:7 t29 p:4c:a t2 p:3c:9 t36 p:4a:a t2 p:2b:a t8 p:26:0 p:35:0 p:39:0 p:3c:0 p:45:0 p:4c:0 t16 p:32:0 t2 p:32:a t29 p:37:9 t3 p:46:a t27 p:3a:a t1 p:43:a t48 p:2d:0 p:32:0 p:35:0 p:37:0 p:39:0 p:3a:0 p:40:0 p:43:0 p:46:0 p:4a:0 p:4c:0 p:51:0 t1 p:48:0 t1 p:2b:0 t1 p:3c:0"
    # revolutionary etude
    ctx = "<start> p:47:e p:4a:e p:4d:d p:4f:d p:53:e t64 p:44:d t17 p:43:d t17 p:41:d t13 p:3e:c t12 p:3f:b t13 p:3e:b t6 p:3e:0 p:3f:0 p:41:0 p:43:0 p:44:0 p:47:0 p:4a:0 p:4d:0 p:4f:0 p:53:0 t4 p:3b:c t8 p:37:b t10 p:38:c t9 p:38:0 p:3b:0 t1 p:37:0 t1 p:37:c t11 p:35:d t9 p:32:b t9 p:35:0 p:37:0 t1 p:33:c t11 p:32:c t11 p:2f:d p:32:0 p:33:0 t9 p:2b:c t11 p:2c:c t6 p:2f:0 t2 p:2b:0 t1 p:2b:c t1 p:2c:0 t11 p:29:c t5 p:2b:0 t4 p:26:b t11 p:27:c t11 p:26:b t1 p:26:0 p:27:0 p:29:0 t11 p:24:d t12 p:1f:c t12 p:24:d t14 p:1f:c t14 p:1f:0 p:24:c p:24:0 p:44:e p:48:c p:4b:e p:4d:d p:50:d t2 p:26:0 t31 p:24:c t27 p:43:e p:4f:e t1 p:1f:b t13 p:1f:0 p:24:0 p:43:0 p:44:0 p:4b:0 p:50:0 t1 p:1f:0 p:48:0 t1 p:24:0 t1 p:4d:0 p:4f:0 t1 p:23:d p:4f:e t1 p:45:d p:4a:e p:4d:e t53 p:44:c t12 p:43:c t3 p:23:0 p:44:0 p:45:0 p:4a:0 p:4d:0 t6 p:43:0 t3 p:41:c t9 p:3e:c t12 p:3f:b t10 p:3e:b t1 p:3e:0 p:3f:0 p:41:0 t9 p:3b:c p:3e:0 t6 p:37:b t10 p:38:c t11 p:37:0 p:38:0 p:3b:0 t1 p:37:c t6 p:4f:0 t5 p:35:c t8 p:32:b t11 p:33:b t2 p:32:0 p:35:0 p:37:0 t8 p:32:b t10 p:2f:d t9 p:2b:c p:2f:0 p:32:0 p:33:0 t11 p:2c:c t10 p:2b:c t2 p:2b:0 p:2c:0 t8 p:29:c t9 p:26:c t6 p:29:0 p:2b:0 t6 p:27:c t3 p:26:0 t7 p:26:a t8 p:27:0 t4 p:24:d t12 p:1f:b t10 p:24:c t15 p:1f:c t13 p:1f:0 p:24:c p:24:0 p:44:e p:4b:d p:50:e t1 p:4d:d t10 p:1f:0 t1 p:1f:a t19 p:18:c p:24:d t26 p:1f:c p:43:e p:4f:e t18 p:18:0 p:1f:0 p:24:0 p:26:0 p:43:0 p:44:0 p:4b:0 p:4f:0 p:50:0 t2 p:4d:0 t4 p:23:d p:4d:e p:4f:e p:51:e p:56:e p:59:f t67 p:5c:e t1 p:50:d t17 p:4f:d p:5b:e t12 p:23:0 p:4f:0 p:50:0 p:51:0 p:56:0 p:59:0 p:5b:0 p:5c:0 t3 p:4d:0 t1 p:4d:d p:59:e t13 p:4a:b t1 p:56:d t10 p:4b:c t2 p:57:d t12 p:4a:c p:56:c t4 p:4a:0 p:4a:0 p:4b:0 p:4d:0 p:56:0 p:57:0 p:59:0 t3 p:53:d t1 p:47:d t11 p:43:b p:4f:c t9 p:44:b p:50:d t9 p:4f:a t1 p:43:b t1 p:43:0 p:4f:0 t1 p:44:0 p:47:0 p:50:0 p:53:0 t1 p:56:0 t6 p:4d:d t1 p:41:c t10 p:3e:b p:4a:c t9 p:3f:b p:4b:c t6 p:3e:0 p:3f:0 p:41:0 p:43:0 p:4d:0 p:4f:0 t1 p:4b:0 t4 p:4a:0 t1 p:3e:b p:4a:c t8 p:47:d t1 p:3b:c t10 p:37:b p:43:c t8 p:37:0 p:3b:0 p:43:0 p:47:0 p:4a:0 t1 p:3e:0 t1 p:38:b p:44:c t7 p:38:0 p:44:0 t4 p:37:b p:43:b t8 p:41:c t1 p:35:c t10 p:32:b p:3e:c t5 p:35:0 p:37:0 p:41:0 p:43:0 t4 p:32:0 p:33:a p:3e:0 p:3f:c t10 p:32:b p:3e:a t8 p:3b:d t1 p:2f:c t10 p:2b:b p:2f:0 p:32:0 p:33:0 p:37:c p:3b:0 p:3e:0 p:3f:0 t9 p:2b:0 t1 p:37:0 p:38:c t1 p:2c:b t9 p:2b:b t1 p:37:b t11 p:29:b p:34:b p:35:b t9 p:26:c p:32:b t5 p:29:0 p:2b:0 p:2c:0 p:34:0 p:35:0 p:37:0 p:38:0 t1 p:26:0 p:32:0 t2 p:27:b t2 p:33:b t6 p:27:0 t4 p:33:0 t1 p:26:a t1 p:32:a t8 p:26:0 t1 p:32:0 t1 p:24:a p:30:b t7 p:24:0 p:30:0 t2 p:23:b p:2f:b t16 p:2b:a p:37:b t11 p:29:a p:35:a t8 p:33:b t1 p:27:a t7 p:26:a t1 p:32:b t11 p:33:c t1 p:27:b t5 p:26:b t2 p:32:a t8 p:24:b p:30:c t3 p:23:0 p:26:0 p:27:0 p:29:0 p:2b:0 p:2f:0 p:32:0 p:33:0 p:35:0 p:37:0 t1 p:24:0 t2 p:23:c t3 p:30:0 t1 p:2f:b t1 p:23:0 t4 p:2f:0 t7 p:2e:b p:3a:d t10 p:2c:c p:38:c t10 p:2b:b p:37:c t9 p:29:b p:35:d t9 p:2b:c p:37:c t10 p:29:a p:35:a t8 p:27:c p:33:d t8 p:32:d t5 p:27:0 p:29:0 p:2b:0 p:2c:0 p:32:0 p:33:0 p:35:0 p:37:0 p:38:0 p:3a:0 t1 p:2e:0 t1 p:2b:0 t7 p:33:c p:3f:d t12 p:32:b p:3e:c t9 p:30:c p:3c:d t11 p:2f:b p:3b:d t5 p:30:0 p:32:0 p:33:0 p:3c:0 p:3e:0 p:3f:0 t6 p:3b:0 t1 p:2f:0 p:30:c p:3c:d t10 p:2f:b t1 p:3b:d t9 p:2c:b p:38:d t10 p:2b:c p:37:d t11 p:2b:0 p:2c:0 p:2f:0 p:37:0 p:38:0 p:3b:0 t1 p:2c:c p:38:c t1 p:30:0 p:3c:0 t7 p:2a:c p:2c:0 t2 p:38:0 t1 p:37:b t14 p:29:b p:2a:0 p:35:c t1 p:37:0 t9 p:27:c p:33:d t16 p:29:c p:35:c t16 p:27:c p:33:e t19 p:30:e t3 p:24:e t2 p:27:0 p:29:0 p:33:0 p:33:0 p:35:0 t3 p:29:0 t1 p:27:0 p:35:0 t9 p:30:0 t21 p:2b:c t13 p:30:b t13 p:32:b t10 p:33:d t11 p:37:c t10 p:3c:d t10 p:3e:d t11 p:3f:d t10 p:3e:c t11 p:3c:c t13 p:37:c t9 p:33:c t10 p:32:b t12 p:30:c t10 p:2b:c t15 p:24:d p:24:0 p:2b:0 p:2b:0 p:30:0 p:32:0 p:33:0 p:37:0 p:3c:0 p:3e:0 p:3f:0 t15 p:2b:c t14 p:30:b t12 p:32:b t13 p:33:c t11 p:32:a t12 p:30:c t14 p:2b:c t7 p:2b:0 p:30:0 p:32:0 p:33:0 t3 p:32:0 t1 p:24:0 t5 p:2b:0"
    # beethoven sonata no 8
    #ctx = "<start> p:24:d p:27:d p:2b:d p:30:d p:33:d p:37:d p:3c:d p:3f:d t1 p:3f:0 t125 t125 t125 t102 p:24:0 p:27:0 p:2b:0 p:30:0 p:33:0 p:37:0 p:3c:0 t1 p:33:a p:3c:a t1 p:37:9 t33 p:3e:a t1 p:32:a p:3b:9 t1 p:37:8 t5 p:33:0 p:37:0 p:3c:0 t125 t4 p:30:b p:32:0 p:37:a p:37:0 p:3c:a p:3e:0 p:3f:b t4 p:3b:0 t41 p:3c:a p:3f:9 t1 p:36:9 p:39:9 t5 p:3c:0 p:3f:0 t1 p:37:0 t40 p:30:0 t125 t125 t4 p:3e:7 t5 p:36:0 p:39:0 p:3f:0 t1 p:3b:5 t1 p:37:6 t9 p:3c:0 t125 t100 p:37:0 t6 p:3b:0 t6 p:3e:0 t13 p:2c:a t61 p:2c:0 p:38:d p:3b:d p:3e:c p:41:d t125 t125 t125 t50 p:38:0 p:3b:0 p:3e:0 p:41:b p:41:0 t1 p:38:9 p:3b:9 p:3e:9 t37 p:3b:9 p:3e:9 p:43:b t1 p:2b:8 p:37:a t3 p:2b:0 p:38:0 p:3b:0 p:3e:0 p:41:0 t125 t2 p:3b:a p:3e:9 p:44:b t1 p:35:a t6 p:3b:0 p:3e:0 t38 p:33:9 p:3b:a p:3e:a p:44:b t2 p:35:0 p:37:0 p:3b:0 p:3e:0 p:43:0 p:44:0 t125 t125 t125 t4 p:43:8 t4 p:3f:6 t1 p:3c:6 t12 p:3b:0 t1 p:3e:0 p:44:0 t125 t101 p:33:0 p:3c:0 p:43:0 t2 p:3f:0 t69 p:33:7 t55 p:3f:c p:42:c p:45:c p:48:c t125 t125 t125 t21 p:45:9 p:48:9 t1 p:42:8 t3 p:3f:8 t33 p:33:0 p:3f:0 p:42:0 p:45:0 p:48:0 t1 p:48:0 t1 p:4a:a t1 p:42:0 t1 p:42:8 p:45:8 p:45:0 t1 p:3e:8 t112 p:4b:b t1 p:42:9 p:45:9 t2 p:3c:9 t1 p:3e:0 p:42:0 p:45:0 t41 p:4b:c t2 p:42:a p:45:a t1 p:3c:9 t5 p:3c:0 p:42:0 p:45:0 p:4b:0 t125 t125 t93 p:4a:8 t4 p:43:6 t2 p:3b:5 t1 p:3c:0 p:3e:5 p:42:0 p:4a:0 p:4b:0 t3 p:45:0 t90 p:3b:0 t1 p:3e:0 t4 p:43:0 p:4a:0 t10 p:3c:8 p:3f:b p:45:a p:48:a t1 p:42:9 t13 p:3c:0 t24 p:3f:0 p:42:a p:42:0 p:45:0 p:48:0 p:4a:a t1 p:3e:a p:45:a t124 p:4b:c t1 p:3c:9 p:42:a p:45:a t49 p:4b:d t1 p:3c:b p:42:b p:45:b t4 p:3c:0 p:3e:0 p:42:0 p:45:0 p:4b:0 t10 p:4a:0 t125 t125 t68 p:4a:9 t4 p:43:8 t1 p:3b:8 t1 p:3e:7 t19 p:3c:0 p:42:0 p:4b:0 t3 p:45:0 t122 p:4c:c t2 p:3a:a t1 p:3c:a p:40:9 p:43:9 p:48:a t125 t37 p:4d:d t2 p:38:b p:3c:b p:41:b p:44:b p:48:b t6 p:3a:0 p:3b:0 p:3c:0 p:3e:0 p:40:0 p:43:0 p:48:0 p:4a:0 p:4c:0 t125 t48 p:38:0 p:41:0 p:44:0 p:48:0 p:4d:0 t2 p:3c:0 t23 p:50:d t125 t29 p:22:b p:2e:a t39 p:4f:b t41 p:50:c t32 p:52:c t22 p:50:b t22 p:4f:b t19 p:4d:b t15 p:4b:b t18 p:4a:b t7 p:4d:9 t10 p:48:a t18 p:46:a t17 p:44:b t21 p:43:b t18 p:41:b t17 p:3f:b t18 p:3e:a t16 p:41:a t20 p:44:a t19 p:41:8 t13 p:44:8 t9 p:3e:9 t44 p:3f:9 t4 p:27:6 t3 p:22:0 p:3e:0 p:3e:0 p:3f:0 p:41:0 p:41:0 p:43:0 p:44:0 p:4d:0 p:4f:0 p:50:0 p:52:0 t1 p:4a:0 p:4b:0 t2 p:44:0 p:48:0 t1 p:2e:0 t1 p:46:0 t77 p:37:7 t1 p:33:6 p:3a:8 t77 p:33:0 p:4b:a t1 p:3a:0 p:3f:0 t1 p:33:7 p:37:0 p:3a:7 p:3f:8 t1 p:37:6 t67 p:33:6 p:37:8 p:3a:6 t64 p:37:7 p:3a:8 t2 p:33:7 t65 p:37:8 p:3a:9 t1 p:33:7 t29 p:3f:a p:4b:c t44 p:41:b p:4d:c t1 p:3a:8 t1 p:33:7 p:37:8 t63 p:37:9 p:3a:9 t1 p:33:9 t33 p:43:b p:4f:c t47 p:43:a p:4f:b t1 p:33:8 p:3c:9 t1 p:38:8 t7 p:27:0 p:33:0 p:37:0 p:3a:0 p:3a:0 p:3f:0 p:41:0 p:43:0 p:4b:0 p:4b:0 p:4d:0 p:4f:0 t1 p:37:0 t11 p:3f:0 t56 p:3c:8 t2 p:33:7 p:38:7 t11 p:33:0 p:38:0 p:3c:0 t57 p:41:8 p:4d:9 t3 p:3c:8 t2 p:33:7 t1 p:38:7 t2 p:33:0 p:38:0 p:3c:0 p:4f:0 t3 p:43:0 t62 p:3c:8 t4 p:33:8 p:38:6 t28 p:41:d t40 p:41:d t1 p:3b:d t1 p:38:d t1 p:32:d t1 p:26:d t5 p:33:0 p:38:0 p:3c:0 p:41:0 t5 p:4d:0 t17 p:35:a t39 p:32:a t26 p:35:b p:38:c p:3b:c p:41:c t42 p:3c:c p:41:c t1 p:35:c p:38:c t1 p:24:c p:30:d t3 p:26:0 p:32:0 p:35:0 p:38:0 p:3b:0 p:3c:0 p:41:0 t16 p:35:0 t1 p:35:a t43 p:24:a p:30:a t30 p:38:c p:3c:b p:41:b t1 p:24:b p:30:b p:35:c t43 p:3e:c t4 p:2f:b p:35:9 p:38:a t1 p:23:a t1 p:24:0 p:30:0 p:35:0 p:38:0 p:3c:0 p:41:0 t84 p:38:a t1 p:35:8 t1 p:32:8 t24 p:23:8 t68 p:23:0 p:2f:0 p:32:0 p:35:0 p:38:0 p:3e:0 p:41:0"

    out_filename = f'performance_{datetime.datetime.now().strftime("%Y%m%d-%H%M%S")}.txt'
    out_file = open(out_filename, 'w')
    out_file.write('')

    state = None
    tokens = tokenizer.encode(ctx).ids
    
    midi_state = None
    token_history = tokens.copy()
    def callback(note: str):
        global midi_state
        for msg, new_state in token_to_midi_message(utils, note, midi_state, end_token_pause=3.0):
            midi_state = new_state
            if msg is not None:
                queue.put(msg)
                #print(msg)
        print(note, end=' ')
        out_file.write(note + ' ')
        out_file.flush()

    # pump context playback
    for id in tokens:
        note = tokenizer.decode([id], skip_special_tokens=False)
        callback(note)

    # sampling parameters
    temperature = 1.0
    top_p = 0.8
    top_k = 0
    repetition_penalty = 1.05  # standard LLM repetition penalty. 1.0 = no penalty
    repetition_view_length = 256  # how far back to look for repetitions
    max_penalty = 1.5  # maximum penalty to apply. 1.0 = no penalty
    decay_factor = 0.99  # how much to decay the penalty by, depending on how far back. 1.0 = no decay
    supress_end = False
    # experimental params
    initial_state_weighting = 0.001  # super experimental. I think this biases long generation towards initial context.
    weighting_period = 0  # phase in and out initial state weighting. mix of creativity and consistency.

    # fast forward through context
    initial_state = None
    ff_tokens = tokens[:-1] if len(tokens) > 1 else []
    ff_chunk_len = 256
    while len(ff_tokens) > 1:
        scores, state = model.forward(ff_tokens[:ff_chunk_len], state)
        # initial state is avg of whole context aa bb pp
        # initial_state = [
        #     s * len(ff_tokens[:ff_chunk_len]) / len(tokens)
        #     if i != 0 and i != 4 # don't touch xx, xx
        #     else s
        #     for i, s in enumerate(state.copy())
        # ] if initial_state != None else state.copy()
        ff_tokens = ff_tokens[ff_chunk_len:]
    tokens = tokens[-1:]
    initial_state = state.copy()

    tokens_generated = 0
    while True:
        if queue.qsize() < max_queue_size:
            scores, state = model.forward(tokens, state)
            if initial_state_weighting > 0:
                weight_factor = initial_state_weighting
                if weighting_period != 0:
                    weight_factor *= max(0, math.cos((2.0 * math.pi) * (tokens_generated % weighting_period) / weighting_period) * 0.5 + 0.5)
                state = [
                    weight_factor * b + (1 - weight_factor) * a  # mix aa, bb, pp
                    if i != 0 and i != 4 # don't touch xx, xx
                    else a
                    for i, (a, b) in enumerate(zip(state, initial_state))
                ]

            scores[pad_token] = -float('inf')
            if supress_end:
                scores[end_token] = -float('inf')
            
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
            tokens_generated += 1
            
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
