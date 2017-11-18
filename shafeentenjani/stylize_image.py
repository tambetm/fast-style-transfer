# Copyright (c) 2016-2017 Shafeen Tejani. Released under GPLv3.
import cv2
from imutils.video import VideoStream
import os

import numpy as np
from os.path import exists
from sys import stdout

import utils
from argparse import ArgumentParser
import tensorflow as tf
import transform

NETWORK_PATH='networks'

def build_parser():
    parser = ArgumentParser()
    parser.add_argument('--network-path', type=str,
                        dest='network_path',
                        help='path to network (default %(default)s)',
                        metavar='NETWORK_PATH', default=NETWORK_PATH)
    return parser

def check_opts(opts):
    assert exists(opts.content), "content not found!"
    assert exists(opts.network_path), "network not found!"

def stylize_and_output(cap, sess, saver, next):
    while(True):
        # Capture frame-by-frame
        #ret, frame = cap.read()
        frame = cap.read()
        
        img_out = frame
        orig_frame = frame
        
        with_style = np.concatenate((img_out, orig_im), axis=0)        
        with_style = pad_im(with_style)
                
        # Display the resulting frame
        cv2.imshow('result', with_style)
        
        if len(rects) > 0:
            if face_start_time == 0:
                face_start_time = time.time()
                style_start_time = time.time()
            show_timer(face_start_time, args.timeout_face, orig_im, default_radius, timer_color, False)
            # Timeout passes
            if time.time() - face_start_time > args.timeout_face:
                img_old = np.swapaxes(orig_frame, 0, 1)                     
                img_old = np.concatenate((img_old, orig_im), axis=0)
                img_old = pad_im(img_old)
                if not args.stylize_preview:
                    cv2.imshow('result', img_old)
                    cv2.waitKey(1)

                stylized_im = stylize_frame(orig_frame)
                img_out = np.swapaxes(stylized_im, 0, 1)                     
                with_style = np.concatenate((img_out, orig_im), axis=0)           
                with_style = pad_im(with_style)

                if args.stylize_preview:
                    for f in np.arange(0., 1.05, 0.05):
                        img = 255. * f + with_style * (1 - f)
                        cv2.imshow('result', img.astype(np.uint8))
                        cv2.waitKey(1)
                    for f in np.arange(0., 1.05, 0.05):
                        img = with_style * f + 255. * (1 - f)
                        cv2.imshow('result', img.astype(np.uint8))
                        cv2.waitKey(10)
                else:
                    for f in np.arange(0, 1.05, 0.05):
                        img = with_style * f + img_old * (1 - f)
                        cv2.imshow('result', img.astype(np.uint8))
                        cv2.waitKey(20)                            
                
                               
                next = (next + 1) % len(styles)
                saver.restore(sess, "./models/"+styles[next]+".ckpt")
                orig_im = read_orig_image(next)
                face_start_time = 0
                style_start_time = time.time()
        else:
            if args.detect_faces:
                face_start_time = 0
                clear_timer(orig_im, default_radius)

        key = cv2.waitKey(10)
        if key == ord('d') or time.time() - style_start_time > args.timeout_style:
                next = (next + 1) % len(styles)
                orig_im = read_orig_image(next)
                saver.restore(sess, "./models/"+styles[next]+".ckpt")
                style_start_time = time.time()
        if key == ord('a'):
                next = (next - 1) % len(styles)-1
                orig_im = read_orig_image(next)
                saver.restore(sess, "./models/"+styles[next]+".ckpt")
                style_start_time = time.time()
        if key & 0xFF == ord('q'):
                break

    # When everything done, release the capture
    #cap.release()
    cap.stop()
    sess.close()
    cv2.destroyAllWindows()

def main():
    parser = build_parser()
    options = parser.parse_args()
    # check_opts(options)

    network = options.network_path
    if not os.path.isdir(network):
        parser.error("Network %s does not exist." % network)

    cap = VideoStream(0).start()
    content_image = cap.read()
    reshaped_content_image = np.ndarray.reshape(content_image, (1,) + content_image.shape)

    with tf.Session() as sess:
        img_placeholder = tf.placeholder(tf.float32, shape=reshaped_content_image.shape,
                                         name='img_placeholder')

        network = transform.net(img_placeholder)
        saver = tf.train.Saver()

        ckpt = tf.train.get_checkpoint_state(options.network_path)

        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
        else:
            raise Exception("No checkpoint found...")
        
        
        while(True):
            content_image = cap.read()
            
            reshaped_content_image = np.ndarray.reshape(content_image, (1,) + content_image.shape)

            prediction = sess.run(network, feed_dict={img_placeholder:reshaped_content_image})[0]
            
            cv2.imshow('result', np.clip(prediction, 0, 255).astype(np.uint8))

            if cv2.waitKey(10) & 0xFF == ord('q'):
                break


    

    # When everything done, release the capture
    #cap.release()
    cap.stop()
    sess.close()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
