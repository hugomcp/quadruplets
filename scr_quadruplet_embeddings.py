import sys
import os
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import cv2
from random import shuffle, sample
import pickle
import sklearn.metrics as mt
import models as md
import math
import datetime
import csv
import argparse
import warnings
from scipy.spatial.distance import cdist
from mpl_toolkits.mplot3d import axes3d

############################################################################################
#	ARGUMENT PARSING
############################################################################################

ap = argparse.ArgumentParser()

ap.add_argument('-d', '--dataset', required=True, help='CSV dataset file')
ap.add_argument('-i', '--input_folder', required=True, help='Data input folder')
ap.add_argument('-o', '--output_folder', required=True, help='Results/debug output folder')
ap.add_argument('-b', '--batch_size', type=int, default=100, help='Learning batch size')
ap.add_argument('-l', '--learning_rate', type=float, default=1e-3, help='Learning rate')
ap.add_argument('-e', '--epochs', type=int, default=10, help='Tot. epochs')
ap.add_argument('-p', '--patience', type=int, default=0, help='Tot. epochs without improvement to stop')
ap.add_argument('-s', '--image_size', type=int, default=64, help='Image size')
ap.add_argument('-m', '--manifold_dimension', type=int, default=128, help='Manifold output dimension')
ap.add_argument('-a', '--alpha', type=float, default=1.0, help='Alpha penalty')
ap.add_argument('-f', '--features', default='1', help='Input features used to define targets')
ap.add_argument('-n', '--features_manifold', default='0', help='Input features manifold flags')


args = vars(ap.parse_args())
FEATURES = args['features'].split(',')
for i, f in enumerate(FEATURES):
    FEATURES[i] = int(f)

FEATURES_MANIFOLD = args['features_manifold'].split(',')
for i, f in enumerate(FEATURES_MANIFOLD):
    FEATURES_MANIFOLD[i] = int(f)

if not os.path.isdir(args['output_folder']):
    os.mkdir(args['output_folder'])

date_time_folder = os.path.join(args['output_folder'], datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S"))
os.mkdir(date_time_folder)

imdb_file = args['dataset'][:-4] + '.dat'

lr_decay = 0.9
decay_epochs = 100

COLORS = ['tab:red', 'tab:green', 'tab:blue', 'tab:orange', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray', 'tab:cyan']

# #####################################################################################################################
# Load Data in '.csv' format: filename, class__1, class_2, ... learning_test_set_flag


# #######################################
# Divide data into learning | test samples
# #######################################
learning_samples = []
test_samples = []
with open(args['dataset']) as f:
    csv_file = csv.reader(f, delimiter=',')
    for row in csv_file:
        row_aux = [row[i] for i in [0] + FEATURES]
        if int(row[-1]) == 0:
            learning_samples.append(row_aux)
        else:
            test_samples.append(row_aux)

for el in learning_samples:
    for i in range(1, len(el)):
        el[i] = int(el[i])

for el in test_samples:
    for i in range(1, len(el)):
        el[i] = int(el[i])

shuffle(learning_samples)

# #######################################
# Create "learning_classes" and "test_classes" information
# #######################################

source_labels_per_column = []
tot_columns = len(FEATURES)
for i in range(1, tot_columns + 1):
    elts = [x[i] for x in learning_samples] + [x[i] for x in test_samples]
    elts = list(set(elts))
    source_labels_per_column.append(elts)

learning_classes = []
for row in learning_samples:
    row[0] = os.path.join(args['input_folder'], row[0])
    learning_classes.append(row[1:])

test_classes = []
for row in test_samples:
    row[0] = os.path.join(args['input_folder'], row[0])
    test_classes.append(row[1:])

classes = np.unique(np.asarray(learning_classes + test_classes), axis=0)
if len(classes.shape) == 1:
    classes = classes[:, np.newaxis]

TOT_FEATURES = len(learning_samples[0]) - 1

labels_per_column = []
for i in range(classes.shape[1]):
    labels = [el[i] for el in classes]
    labels_per_column.append(list(np.unique(np.asarray(labels))))

#####################################################################################################

# get the image size
img = cv2.imread(learning_samples[0][0], cv2.IMREAD_COLOR)
HEIGHT_IMGS = args['image_size']
WIDTH_IMGS = args['image_size']
DEPTH_IMGS = np.size(img, 2)


############################
# AUXILIARY FUNCTIONS

def read_batch(l_s, rand_idx, imd, manifold_columns):
    cls = np.zeros((len(rand_idx), tot_columns), dtype=np.float32)
    imgs = np.zeros((len(rand_idx), HEIGHT_IMGS, WIDTH_IMGS, DEPTH_IMGS), dtype=np.float32)

    for i, idx in enumerate(rand_idx):
        imgs[i, :, :, :] = cv2.resize(cv2.imread(l_s[idx][0], cv2.IMREAD_COLOR), (HEIGHT_IMGS, WIDTH_IMGS))
        for j in range(tot_columns):
            cls[i, j] = l_s[idx][j + 1]

    alphas = np.zeros((len(rand_idx), len(rand_idx), len(manifold_columns)), dtype=np.float32)
    for i in range(len(manifold_columns)):
        cur_cls = cls[:, i]
        cur_pair = mt.pairwise_distances(cur_cls.reshape(-1, 1))
        if manifold_columns[i] == 0:
            cur_pair[cur_pair > 0] = 1.0
        alphas[:, :, i] = cur_pair

    alphas = np.sum(alphas, axis=2)
    
    alphas[alphas > 1] = 1
    alphas = np.reshape(alphas, [-1, 1])

    for i, _ in enumerate(rand_idx):
        imgs[i, :, :, :] = np.divide(imgs[i, :, :, :], imd)
    return imgs, alphas


def get_imdb(paths):
    imdb = np.zeros((HEIGHT_IMGS, WIDTH_IMGS, DEPTH_IMGS), dtype=np.float32)
    for i, pt in enumerate(paths):
        print('IMDB: {}/{}'.format(i, len(paths)))
        img = cv2.resize(cv2.imread(pt[0], cv2.IMREAD_COLOR), (HEIGHT_IMGS, WIDTH_IMGS))
        imdb = imdb + img
    imdb = imdb / len(paths)
    return imdb


########################################################################################################################
# TENSORFLOW
########################################################################################################################

###############################
# Create Placeholders

x_input_shape = (None, HEIGHT_IMGS, WIDTH_IMGS, DEPTH_IMGS)
x_inputs = tf.placeholder(tf.float32, shape=x_input_shape)
alphas = tf.placeholder(tf.float32, shape=None)
generation_num = tf.Variable(0, trainable=False)


def distance_rows_matrix(A):
    r = tf.reduce_sum(A * A, 1)
    r = tf.reshape(r, [-1, 1])
    D = r - 2 * tf.matmul(A, tf.transpose(A)) + tf.transpose(r)
    return D


def pairwise_difference_elements_array(a):
    a = tf.reshape(a, [-1])
    aux = tf.reshape(tf.tile(a, [tf.size(a)]), [tf.size(a), -1])
    diff = tf.transpose(aux) - aux
    return tf.reshape(diff, [-1])


def pairwise_sum_elements_array(a):
    a = tf.reshape(a, [-1])
    aux = tf.reshape(tf.tile(a, [tf.size(a)]), [tf.size(a), -1])
    diff = tf.transpose(aux) - aux
    return tf.reshape(diff, [-1])



def loss_quadruplet(logits, difs_al):
    difs_al = tf.reshape(difs_al, [-1])

    # difs_al = tf.Print(difs_al, ["DIFS_AL ", tf.shape(difs_al), "=", difs_al], summarize=50)
    # logits = tf.Print(logits, ["LOGITS ", tf.shape(logits), "=", logits], summarize=50)

    dists = tf.reshape(distance_rows_matrix(logits), [-1])

    # dists = tf.Print(dists, ["DISTS ", tf.shape(dists), "=", dists], summarize=50)

    dists_2 = pairwise_difference_elements_array(dists)
    difs_al_2 = pairwise_difference_elements_array(difs_al)

    # dists_2 = tf.Print(dists_2, ["DISTS_2 ", dists_2], summarize=100)
    # difs_al_2 = tf.Print(difs_al_2, ["DIFS_AL_2 ", difs_al_2], summarize=100)

    difs_al_21 = tf.sign(difs_al_2)

    lo = tf.multiply(-tf.sign(difs_al_21), dists_2 - difs_al_21 * tf.constant(args['alpha']))

    # lo = tf.Print(lo, ["LO ", lo], summarize=100)

    lo = tf.math.maximum(lo, 0.0)

    # lo = tf.Print(lo, ["MAX(LO, 0) ", lo], summarize=100)

    lo = tf.reduce_sum(lo)

    # lo = tf.Print(lo, ["OUT LOSS ", lo])

    return lo


debug_logits = tf.placeholder(tf.float32, shape=(None, 3))
debug_al = tf.placeholder(tf.float32, shape=None)
debug_loss = loss_triplet(debug_logits, debug_al)

with tf.variable_scope('model_definition') as scope:
    model_outputs = md.vgg(x_inputs, 1.0, args['manifold_dimension'])
    #model_outputs = md.resnet(x_inputs, 1.0, args['manifold_dimension'])
    scope.reuse_variables()

loss = loss_quadruplet(model_outputs, alphas))

model_learning_rate = tf.train.exponential_decay(args['learning_rate'], generation_num, decay_epochs, lr_decay,
                                                 staircase=True)
my_optimizer = tf.train.GradientDescentOptimizer(model_learning_rate)
gradients = my_optimizer.compute_gradients(loss)
train_op = my_optimizer.apply_gradients(gradients)

sess = tf.Session()
plt.ion()

saver = tf.train.Saver()

########################################################################################################################
# LEARNING MAIN ()
########################################################################################################################

init = tf.global_variables_initializer()
sess.run(init)

if not os.path.isfile(imdb_file):
    imdb = get_imdb(learning_samples)
    with open(imdb_file, 'wb') as f:
        pickle.dump(imdb, f)
else:
    with open(imdb_file, 'rb') as f:
        imdb = pickle.load(f)

train_loss = []

for e in range(args['epochs']):

    i = 0
    sess.run(generation_num.assign(e))
    epoch_loss = 0

    while i < len(learning_samples):
        rand_idx = np.random.choice(range(len(learning_samples)), size=args['batch_size'], replace=True)
        # rand_idx = np.asarray(range(i, np.min([i + args['batch_size'], len(learning_samples)])))

        rand_imgs, rand_alphas = read_batch(learning_samples, rand_idx, imdb, FEATURES_MANIFOLD)

        sess.run(train_op, feed_dict={x_inputs: rand_imgs, alphas: rand_alphas})

        t_loss = sess.run(loss, feed_dict={x_inputs: rand_imgs, alphas: rand_alphas})

        print('Learning\tEpoch\t{}/{}\tBatch {}/{}\tLoss={:.5f}'.format(e + 1, args['epochs'],
                                                                        (i + 1) // args['batch_size'] + 1, math.ceil(
                len(learning_samples) / args['batch_size']), t_loss))
        i += args['batch_size']
        epoch_loss += t_loss * len(rand_idx)

    epoch_loss /= len(learning_samples)
    train_loss.append(epoch_loss)

    eval_indices = range(1, e + 2)
    fig_1 = plt.figure(1)
    plt.clf()
    plt.semilogy(eval_indices, train_loss, 'g-o', label='Training')
    plt.xlabel('Generation')
    plt.ylabel('Loss')
    plt.grid(which='major', axis='both')

    fig_1.show()
    plt.pause(0.01)

    ################################################################################################

    if args['patience'] > 0:
        last_difs = np.diff(np.asarray(train_loss))
        if len(last_difs) >= args['patience']:
            if (last_difs[-args['patience']:] > 0).all():
                break

plt.savefig(os.path.join(date_time_folder, 'Learning.png'))

########################################################################################################################
# Save model
########################################################################################################################

saver.save(sess, os.path.join(date_time_folder, 'model'))


#######################################
# Learning + test final state
# #######################################

i = 0
learning_out = []
learning_gt = []
while i < len(learning_samples):
    rand_idx = np.asarray(range(i, np.min([i + args['batch_size'], len(learning_samples)])))

    rand_imgs, _ = read_batch(learning_samples, rand_idx, imdb, FEATURES_MANIFOLD)
    y_out = sess.run(model_outputs, feed_dict={x_inputs: rand_imgs})

    learning_out.extend(y_out)
    learning_gt.extend([learning_classes[idx] for idx in rand_idx])
    i += args['batch_size']

i = 0
test_out = []
test_gt = []
while i < len(test_samples):
    rand_idx = np.asarray(range(i, np.min([i + args['batch_size'], len(test_samples)])))

    rand_imgs, _ = read_batch(test_samples, rand_idx, imdb, FEATURES_MANIFOLD)
    y_out = sess.run(model_outputs, feed_dict={x_inputs: rand_imgs})

    test_out.extend(y_out)
    test_gt.extend([test_classes[idx] for idx in rand_idx])
    i += args['batch_size']

learning_out = np.asarray(learning_out)
learning_gt = np.asarray(learning_gt)

test_out = np.asarray(test_out)
test_gt = np.asarray(test_gt)

# #######################################
# Write learning/test scores to file
# #######################################
file_out = open(os.path.join(date_time_folder, 'scores_learning.txt'), "a+")
for i in range(len(learning_samples)):
    splited = os.path.split(learning_samples[i][0])
    file_out.write("%s, " % splited[1])
    for j in range(1, len(learning_samples[i])):
        file_out.write("%d, " % learning_samples[i][j])
    for j in range(args['manifold_dimension']):
        file_out.write("%f" % learning_out[i][j])
        if j < args['manifold_dimension']-1:
            file_out.write(", ")
        else:
            file_out.write("\n")
file_out.close()

file_out = open(os.path.join(date_time_folder, 'scores_test.txt'), "a+")
for i in range(len(test_samples)):
    splited = os.path.split(test_samples[i][0])
    file_out.write("%s, " % splited[1])
    for j in range(1, len(test_samples[i])):
        file_out.write("%d, " % test_samples[i][j])
    for j in range(args['manifold_dimension']):
        file_out.write("%f" % test_out[i][j])
        if j < args['manifold_dimension']-1:
            file_out.write(", ")
        else:
            file_out.write("\n")
file_out.close()


# #######################################
# Closest instance performance measures
# #######################################

dists = cdist(test_out, learning_out)

# find the rank of the closest learning instance of the corresponding class (per feature)
rank_best_same_class = []
for i in range(len(test_samples)):
    idx = np.argsort(dists[i, :])
    aux = [0] * TOT_FEATURES
    for j in range(TOT_FEATURES):
        un_ids, un_idx = np.unique([learning_gt[k][j] for k in idx], return_index=True, axis=0)
        idx_same_class = np.asarray([k for k in range(len(un_ids)) if (un_ids[k] == test_gt[i][j])])
        aux[j] = 1 + sum(un_idx < un_idx[idx_same_class])
    rank_best_same_class.append(aux)

acc_rank_best_same_class = []
for j in range(TOT_FEATURES):
    aux = []
    tot_classes_label = len(labels_per_column[j])
    ranks = [el[j] for el in rank_best_same_class]
    for i in range(1, tot_classes_label + 1):
        acc_val = sum(np.asarray(ranks) <= i) / len(ranks)
        aux.append(acc_val)
    acc_rank_best_same_class.append(aux)

    fig = plt.figure(2 + j)
    plt.plot(range(1, len(acc_rank_best_same_class[j]) + 1), acc_rank_best_same_class[j], 'go-')
    plt.title('Retrieval_%d.png' % (j))
    plt.xlabel('Rank')
    plt.ylabel('Acc Best')
    plt.ylim([0, 1])
    plt.xlim([1, len(acc_rank_best_same_class[j])])
    plt.grid(which='major', axis='both')
    plt.show()
    plt.pause(0.01)
    plt.savefig(os.path.join(date_time_folder, 'Retrieval_%d.png' % (j)))

file_out = open(os.path.join(date_time_folder, 'configs.txt'), "a+")
for k in args.keys():
    file_out.write('%s: %s\n' % (k, args[k]))
file_out.close()

file_out = open(os.path.join(date_time_folder, 'results.txt'), "a+")
for i in range(TOT_FEATURES):
    print('Feature {}'.format(i + 1))
    AUC = sum(acc_rank_best_same_class[i]) / len(acc_rank_best_same_class[i])
    print("\tRank-1 {:.4f}".format(acc_rank_best_same_class[i][0]))
    file_out.write("\tRank-1 %.4f\n" % acc_rank_best_same_class[i][0])
    print("\tAUC {:.4f}".format(AUC))
    file_out.write("\tAUC %.4f\n" % AUC)

file_out.close()


fig = plt.figure(2 + TOT_FEATURES)
if args['manifold_dimension'] == 2 or args['manifold_dimension'] == 3:
    if args['manifold_dimension'] == 2:
        ax = fig.add_subplot(111)
    else:
        ax = fig.add_subplot(111, projection='3d')
    for line, p in enumerate(classes):
        idx = np.where((learning_gt == p).all(axis=1))[0]
        if len(idx) > 0:
            pts = [learning_out[i] for i in idx]
            ax.scatter(*zip(*pts), c=COLORS[line % len(COLORS)], marker='+')


plt.grid(b=True, which='both', axis='both')
fig.show()
plt.pause(0.01)
plt.savefig(os.path.join(date_time_folder, 'Feature_Space.png'))




fig = plt.figure(3 + TOT_FEATURES)
if args['manifold_dimension'] == 2 or args['manifold_dimension'] == 3:
    if args['manifold_dimension'] == 2:
        ax = fig.add_subplot(111)
    else:
        ax = fig.add_subplot(111, projection='3d')
    for line, p in enumerate(classes):
        idx = np.where((learning_gt == p).all(axis=1))[0]
        label_p = ''.join([str(pp)+',' for pp in p[1:]])
        if len(idx) > 0:
            pts = [learning_out[i] for i in idx]
            cls = [learning_gt[i][0] for i in idx]
            centroid = np.mean(np.asarray(pts), axis=0)
            if args['manifold_dimension'] == 2:
                ax.scatter(*zip(*pts), c=COLORS[line % len(COLORS)], marker='+')
                ax.text(centroid[0], centroid[1], label_p)
            if args['manifold_dimension'] == 3:
                ax.scatter(*zip(*pts), c=COLORS[line % len(COLORS)], marker='+')
                ax.text(centroid[0], centroid[1], centroid[2], label_p)

plt.grid(b=True, which='both', axis='both')
fig.show()
plt.pause(0.01)
plt.savefig(os.path.join(date_time_folder, 'Feature_Space_pts.png'))


name = input('Close app?')
