import sys
sys.path.insert(0, 'src')
import transform
import numpy as np
import tensorflow as tf
import cv2
import argparse
import time
import dlib
import math
from src.client import send_image

def setup_parser():
    """Options for command-line input."""
    parser = argparse.ArgumentParser(description="""Use a trained fast style
                                     transfer model to filter webcam feed.""")
    parser.add_argument('--model_path',
                        default='./models/starry_final.ckpt',
                        help='Path to .ckpt for the trained model.')
    parser.add_argument('--upsample_method',
                        help="""The upsample method that was used to construct
                        the model being loaded. Note that if the wrong one is
                        chosen an error will be thrown.""",
                        choices=['resize', 'deconv'],
                        default='resize')
    parser.add_argument('--resolution',
                        help="""Dimensions for webcam. Note that, depending on
                        the webcam, only certain resolutions will be possible.
                        Leave this argument blank if want to use default
                        resolution.""",
                        nargs=2,
                        type=int,
                        default=None)
    parser.add_argument('--capture_device', type=int, default=0)
    parser.add_argument('--fullscreen', action="store_true", default=False)
    parser.add_argument('--vertical', action="store_true", default=False)
    parser.add_argument('--timeout', help='How many seconds to wait before switching to next style', default=5)
    parser.add_argument('--timeout_qr', help='How many seconds to show output image with qr', default=10)
    parser.add_argument('--server_url', help='Server url for uploading images', default="http://miranda.rm.mt.ut.ee:5000/uploadImage")
    parser.add_argument('--stylize_preview', action="store_true", default=False)
    return parser

def read_orig_image(index):
        orig_im = cv2.imread("./styles/"+styles[index]+".jpg")
        factory = 240. / orig_im.shape[0]
        factorx = 240. / orig_im.shape[1]
        factor = min(factorx, factory)
        orig_im = cv2.resize(orig_im, (0, 0), fx=factor, fy=factor, interpolation=cv2.INTER_AREA)
        orig_im = np.pad(orig_im, ((y_new - 400 - orig_im.shape[0] + 30, 0), (0, x_new - orig_im.shape[1]), (0,0)), 'constant')
        text_size_ln1 = cv2.getTextSize(titles[index],cv2.FONT_HERSHEY_SIMPLEX,1,0)[0];
        text_size_ln2 = cv2.getTextSize("by "+authors[index],cv2.FONT_HERSHEY_SIMPLEX,1,0)[0];
        cv2.putText(orig_im, titles[index], (orig_im.shape[1]-text_size_ln1[0], orig_im.shape[0]-(10+2*text_size_ln1[1])), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 1, lineType=cv2.LINE_AA)
        cv2.putText(orig_im, "by "+authors[index], (orig_im.shape[1]-text_size_ln2[0], orig_im.shape[0]-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 1, lineType=cv2.LINE_AA)
        return orig_im
       
# displays clock-similar animation next to original style image 
def show_timer(start_time, timeout, orig_im, radius, color, reverse):
    center = (orig_im.shape[1]-(radius+3), 30+radius)
    if reverse:
        cv2.circle(orig_im, center, radius, (0,0,0), thickness=-1, lineType=cv2.LINE_AA)
        cv2.circle(orig_im, center, radius, color, thickness=1, lineType=cv2.LINE_AA)
        cv2.ellipse(orig_im, center, (radius, radius), -90, 0, 360 - 360/timeout*math.floor(time.time() - start_time), color, -1)
    else:
        cv2.circle(orig_im, center, radius, color, thickness=1, lineType=cv2.LINE_AA)
        cv2.ellipse(orig_im, center, (radius, radius), -90, 0, 360/timeout*math.floor(time.time() - start_time), color, -1)
        
        
def clear_timer(orig_im, radius):
    center = (orig_im.shape[1]-(radius+3), 30+radius)
    radius += 3
    cv2.circle(orig_im, center, radius, (0,0,0), thickness=-1, lineType=cv2.LINE_AA)

def pad_im(img):
    padx = (540 - img.shape[1]) // 2
    pady = (960 - img.shape[0]) // 2
    return np.pad(img, ((pady, pady), (padx, padx), (0, 0)), "constant")

def add_qr(qr_img, dest_img):
    qr_ndarray = np.array(qr_img, dtype=np.float32) * 255
    qr = cv2.cvtColor(qr_ndarray, cv2.COLOR_GRAY2BGR)
    dest_img[30:30+qr.shape[0], 351:(351+qr.shape[1]), :] = qr

def stylize_frame(frame):
    img_4d = frame[np.newaxis, :]
                
    # Our operations on the frame come here
    img_out = sess.run(Y, feed_dict={X: img_4d})
    img_out = np.clip(img_out, 0, 255)
    img_out = np.squeeze(img_out).astype(np.uint8)
    return cv2.cvtColor(img_out, cv2.COLOR_BGR2RGB)

def stylize_and_output(cap, sess, saver, next):
        default_radius = 13
        print('Loading up model...')
        saver.restore(sess, "./models/"+styles[next]+".ckpt")
        print('Begin filtering...')
        # init original style image
        orig_im = read_orig_image(next)
        start_time = 0
        qr_img = None
        timer_color = (100,100,100)
        while(True):
            # Capture frame-by-frame
            ret, frame = cap.read()
            
            img_out = frame
            orig_frame = frame
            
            if args.stylize_preview:
                img_out = stylize_frame(frame)
                   
            if args.vertical:
                frame = np.swapaxes(frame, 0, 1)
                img_out = np.swapaxes(img_out, 0, 1)

            with_style = np.concatenate((img_out, orig_im), axis=0)        
            with_style = pad_im(with_style)
                    
            # Display the resulting frame
            cv2.imshow('result', with_style)

            # If face detected, start countdown to take a picture
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            rects = detector(gray, 0)
            
            if len(rects) > 0:
                if start_time == 0:
                    start_time = time.time()
                show_timer(start_time, args.timeout, orig_im, default_radius, timer_color, False)
                # Timeout passes
                if time.time() - args.timeout > start_time:
                    img_old = np.swapaxes(orig_frame, 0, 1)                     
                    img_old = np.concatenate((img_old, orig_im), axis=0)
                    img_old = pad_im(img_old)
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
                    
                    # Send output image to server
                    clear_timer(orig_im, default_radius)
                    output_im = np.concatenate((img_out, orig_im), axis=0)
                    qr_img = send_image(pad_im(output_im), args.server_url)
                    
                    # Show image with QR and timer                     
                    freeze_start = time.time()                        
                    while(time.time() - args.timeout_qr < freeze_start): 
                        clear_timer(orig_im, default_radius)
                        #show_timer(freeze_start, args.timeout_qr, orig_im, default_radius, timer_color, True)
                        
                        add_qr(qr_img, orig_im)
                        cv2.imshow('result', pad_im(np.concatenate((img_out, orig_im), axis=0)))
                        cv2.waitKey(1000)
                                                    
                    next = (next + 1) % len(styles)
                    saver.restore(sess, "./models/"+styles[next]+".ckpt")
                    orig_im = read_orig_image(next)
                    start_time = 0
            else:
                    start_time = 0
                    clear_timer(orig_im, default_radius)

            key = cv2.waitKey(10)
            if key == ord('d'):
                    next = (next + 1) % len(styles)
                    orig_im = read_orig_image(next)
                    saver.restore(sess, "./models/"+styles[next]+".ckpt")
                    start_time = time.time()
            if key == ord('a'):
                    next = (next - 1) % len(styles)-1
                    orig_im = read_orig_image(next)
                    saver.restore(sess, "./models/"+styles[next]+".ckpt")
                    start_time = time.time()
            if key & 0xFF == ord('q'):
                    break

        # When everything done, release the capture
        cap.release()
        sess.close()
        cv2.destroyAllWindows()
        


if __name__ == '__main__':

    # Command-line argument parsing.
    parser = setup_parser()
    args = parser.parse_args()
    model_path = args.model_path
    upsample_method = args.upsample_method
    resolution = args.resolution

    # Instantiate video capture object.
    cap = cv2.VideoCapture(args.capture_device)

    # Set resolution
    if resolution is not None:
        x_length, y_length = resolution
        cap.set(3, x_length)  # 3 and 4 are OpenCV property IDs.
        cap.set(4, y_length)
    x_new = int(cap.get(3))
    y_new = int(cap.get(4))
    print('Resolution is: {0} by {1}'.format(x_new, y_new))

    # Create the graph.
    g = tf.Graph()
    soft_config = tf.ConfigProto(allow_soft_placement=True)
    soft_config.gpu_options.allow_growth = True
    #soft_config.gpu_options.per_process_gpu_memory_fraction=0.33
    shape = [1, y_new, x_new, 3]

    # init authors, titles and styles
    authors = ["E.Munch", "F.Picabia", "K.Hokusai", "P.Picasso", "L.Afremov", "W.Turner"]
    titles = ["Scream", "Udnie", "The Wave", "La Muse", "Rain Princess", "The Shipwreck"]
    styles = ["scream", "udnie", "wave", "la_muse", "rain_princess", "wreck"]
    
    # Create face detector
    detector = dlib.get_frontal_face_detector()

    if args.vertical:
        t = x_new
        x_new = y_new
        y_new = t

    # open graph
    with g.as_default():
        X = tf.placeholder(tf.float32, shape=shape, name='img_placeholder')
        Y = transform.net(X)

        # restore the model to the session.
        saver = tf.train.Saver()

        if args.fullscreen:
            cv2.namedWindow("result", cv2.WND_PROP_FULLSCREEN)
            cv2.setWindowProperty("result", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

        next = 0
        sess = tf.Session(config=soft_config)

        stylize_and_output(cap, sess, saver, next)
