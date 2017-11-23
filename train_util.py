#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Minhyuk Sung (mhsung@cs.stanford.edu)
# April 2017

import math
import numpy as np
import os
import sys
import tensorflow as tf
import time


def hms_string(sec_elapsed):
    h = int(sec_elapsed / (60 * 60))
    m = int((sec_elapsed % (60 * 60)) / 60)
    s = sec_elapsed % 60.
    return "{}:{:>02}:{:>05.2f}".format(h, m, s)


def evaluate(sess, net, data):
    count = 0
    loss = 0.0
    accuracy = 0.0
    summary = None

    for X, Y, Yp, Z in data:
        # NOTE:
        # Take the summary of the last random batch.
        summary, step_loss, step_accuracy = sess.run([
            net.summary, net.loss, net.accuracy], feed_dict={
                net.X: X,
                net.Y: Y,
                net.Yp: Yp,
                net.Z: Z,
                net.is_training: False})

        count += data.step_size
        loss += (step_loss * data.step_size)
        accuracy += (step_accuracy * data.step_size)

    loss /= float(count)
    accuracy /= float(count)
    return loss, accuracy, summary


def train(sess, net, train_data, val_data, n_epochs, snapshot_epoch,
        model_dir='model', log_dir='log', data_name='',
        output_generator=None):
    train_writer = tf.summary.FileWriter(log_dir + '/train', sess.graph)
    test_writer = tf.summary.FileWriter(log_dir + '/test')

    # Create snapshot directory.
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    print ("\n=================")
    print ("Training started.")

    start_time = time.time()

    for epoch in range(1, n_epochs + 1):
        for X, Y, Yp, Z in train_data:
            step, _, = sess.run([net.global_step, net.train_op], feed_dict={
                net.X: X,
                net.Y: Y,
                net.Yp: Yp,
                net.Z: Z,
                net.is_training: True})

            summary, loss, accuracy = sess.run([
                net.summary, net.loss, net.accuracy], feed_dict={
                    net.X: X,
                    net.Y: Y,
                    net.Yp: Yp,
                    net.Z: Z,
                    net.is_training: False})

            if step > 1000:
                train_writer.add_summary(summary, step)

            elapsed = time.time() - start_time
            msg = " -"
            msg += "" if data_name == '' else " [{}]".format(data_name)
            msg += " Step: {:d}".format(step)
            msg += " | Iter {:d}/{:d}".format(
                    train_data.start, train_data.n_data)
            msg += " | Batch Loss: {:6f}".format(loss)
            msg += " | Batch Accu: {:5f}".format(accuracy)
            msg += " | Elapsed Time: {}".format(hms_string(elapsed))
            print(msg)
            sys.stdout.write("\033[1A[\033[2K")

        # Calculate total train and validation loss and accuracy.
        loss, accuracy, _ = evaluate(sess, net, train_data)
        msg = "||"
        msg += "" if data_name == '' else " [{}]".format(data_name)
        msg += " Epoch: {:d}".format(epoch)
        msg += " | Train Loss: {:6f}".format(loss)
        msg += " | Train Accu: {:5f}".format(accuracy)

        val_loss, val_accuracy, val_summary = evaluate(sess, net, val_data)
        msg += " | Valid Loss: {:6f}".format(val_loss)
        msg += " | Valid Accu: {:5f}".format(val_accuracy)

        if step > 1000:
            test_writer.add_summary(val_summary, step)

        elapsed = time.time() - start_time
        remaining = elapsed / epoch * (n_epochs - epoch)
        msg += " | Elapsed Time: {} | Remaining Time: {} ||".format(
                hms_string(elapsed), hms_string(remaining))
        print(msg)

        if epoch % snapshot_epoch == 0:
            # Save snapshot.
            sys.stdout.write("Saving epoch {:d} snapshot... ".format(epoch))
            net.saver.save(sess, model_dir + '/tf_model.ckpt',
                    global_step=step)
            print('Done.')

            # Generate outputs.
            if output_generator is not None:
                output_generator(sess, 'snapshot_{:06d}'.format(epoch))

    train_writer.close()
    test_writer.close()

    elapsed = time.time() - start_time
    print ("Training finished.")
    print (" - Elapsed Time: {}".format(hms_string(elapsed)))
    print ("Saved '{}'.".format(
        net.saver.save(sess, model_dir + '/tf_model.ckpt', global_step=step)))
    print ("=================\n")

