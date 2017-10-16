import sys
sys.path.insert(0, 'src')
import transform, numpy as np
import cv2
import tensorflow as tf
import argparse

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
    parser.add_argument('--capture_device', default=1)
    parser.add_argument('--fullscreen', action="store_true", default=False)
    parser.add_argument('--vertical', action="store_true", default=False)
    parser.add_argument('--canvas_size', nargs=2, type=int, default=None)
    return parser

def read_orig_image(filename):
	orig_im = cv2.imread(filename)
	factory = 300. / orig_im.shape[0]
	factorx = float(x_new) / orig_im.shape[1]
	factor = min(factorx, factory)
	orig_im = cv2.resize(orig_im, (0, 0), fx=factor, fy=factor, interpolation=cv2.INTER_AREA)
        padx = (x_new - orig_im.shape[1]) // 2
        pady = 10
        orig_im = np.pad(orig_im, ((pady, pady), (padx, x_new - orig_im.shape[1] - padx), (0,0)), 'constant')
	return orig_im

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
    shape = [1, y_new, x_new, 3]

    #styles = ["scream", "udnie", "wave", "la_muse", "rain_princess", "wreck"]
    styles = ["udnie", "wave", "rain_princess", "la_muse"]

    if args.vertical:
        t = x_new
        x_new = y_new
	y_new = t

    with g.as_default():
	X = tf.placeholder(tf.float32, shape=shape, name='img_placeholder')
	Y = transform.net(X)

	# Saver used to restore the model to the session.
	saver = tf.train.Saver()

	if args.fullscreen:
	    cv2.namedWindow("result", cv2.WND_PROP_FULLSCREEN)
	    cv2.setWindowProperty("result", cv2.WND_PROP_FULLSCREEN, cv2.cv.CV_WINDOW_FULLSCREEN)
	models = ["./models/scream.ckpt", "./models/udnie.ckpt", "./models/wave.ckpt", "./models/la_muse.ckpt", "./models/rain_princess.ckpt", "./models/wreck.ckpt"]
	next = 0
	sess = tf.Session(config=soft_config)

	# Begin filtering.
	print('Loading up model...')
	orig_im = read_orig_image("./styles/"+styles[next]+".jpg")

	saver.restore(sess, "./models/"+styles[next]+".ckpt")
	print('Begin filtering...')

	while(True):
	    # Capture frame-by-frame
	    ret, frame = cap.read()

	    # Make frame 4-D
	    img_4d = frame[np.newaxis, :]

	    # Our operations on the frame come here
	    img_out = sess.run(Y, feed_dict={X: img_4d})
	    img_out = np.squeeze(img_out).astype(np.uint8)
	    img_out = cv2.cvtColor(img_out, cv2.COLOR_BGR2RGB)

            if args.vertical:
                frame = np.swapaxes(frame, 0, 1)
		img_out = np.swapaxes(img_out, 0, 1)
	    with_style = np.concatenate((img_out, orig_im), axis=0)

            if args.canvas_size:
	        padx = (args.canvas_size[0] - with_style.shape[1]) // 2
	        pady = (args.canvas_size[1] - with_style.shape[0]) // 2
	        with_style = np.pad(with_style, ((pady, pady), (padx, padx), (0, 0)), "constant")

	    # Display the resulting frame
	    cv2.imshow('result', with_style)
	    key = cv2.waitKey(1)
	    if key == ord('a'):
		if next == len(styles)-1:
		    next = 0
		else:
		    next += 1
		orig_im = read_orig_image("./styles/"+styles[next]+".jpg")
		saver.restore(sess, "./models/"+styles[next]+".ckpt")
	    if key == ord('d'):
		if next == 0:
		    next = len(styles)-1
		else:
		    next -= 1
		orig_im = read_orig_image("./styles/"+styles[next]+".jpg")
		saver.restore(sess, "./models/"+styles[next]+".ckpt")
	    if key & 0xFF == ord('q'):
	        break

	# When everything done, release the capture
	cap.release()
	sess.close()
	cv2.destroyAllWindows()
