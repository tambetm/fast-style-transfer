import sys
sys.path.insert(0, 'src')
import transform, numpy as np
import cv2
import tensorflow as tf
import argparse

# TODO: work out the dimension h/w convention for opencv + neural net.
# TODO: feed appropriate fps to writer.


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
    return parser


if __name__ == '__main__':

    # Command-line argument parsing.
    parser = setup_parser()
    args = parser.parse_args()
    model_path = args.model_path
    upsample_method = args.upsample_method
    resolution = args.resolution

    # Instantiate video capture object.
    cap = cv2.VideoCapture(1)

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
    with g.as_default():
	X = tf.placeholder(tf.float32, shape=shape, name='img_placeholder')
	Y = transform.net(X)

	# Saver used to restore the model to the session.
	saver = tf.train.Saver()

	cv2.namedWindow("result", cv2.WND_PROP_FULLSCREEN)
	cv2.setWindowProperty("result", cv2.WND_PROP_FULLSCREEN, cv2.cv.CV_WINDOW_FULLSCREEN)
	models = ["./models/scream.ckpt", "./models/udnie.ckpt", "./models/wave.ckpt", "./models/la_muse.ckpt", "./models/rain_princess.ckpt", "./models/wreck.ckpt"]
	next = 0
	sess = tf.Session(config=soft_config)

	# Begin filtering.
	print('Loading up model...')
	saver.restore(sess, models[next])
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

	    # Display the resulting frame
	    cv2.imshow('result', img_out)
	    key = cv2.waitKey(1)
	    if key == ord('a'):
		if next == len(models)-1:
		    next = 0
		else:
		    next += 1
		saver.restore(sess, models[next])
	    if key == ord('d'):
		if next == 0:
		    next = len(models)-1
		else:
		    next -= 1
		saver.restore(sess, models[next])
	    if key & 0xFF == ord('q'):
	        break

	# When everything done, release the capture
	cap.release()
	sess.close()
	cv2.destroyAllWindows()
