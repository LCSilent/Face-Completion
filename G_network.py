import tensorflow as tf

with tf.variable_scope('weights'):
    weights = {
        'conv0_1': tf.get_variable('conv0_1', shape=[3, 3, 3, 32], initializer=tf.contrib.layers.xavier_initializer()),
        'conv0_2': tf.get_variable('conv0_2', shape=[3, 3, 32, 32], initializer=tf.contrib.layers.xavier_initializer()),
        'conv1_1': tf.get_variable('conv1_1', shape=[3, 3, 32, 64], initializer=tf.contrib.layers.xavier_initializer()),
        'conv1_2': tf.get_variable('conv1_2', shape=[3, 3, 64, 64], initializer=tf.contrib.layers.xavier_initializer()),
        'conv2_1': tf.get_variable('conv2_1', shape=[3, 3, 64, 128],
                                   initializer=tf.contrib.layers.xavier_initializer()),
        'conv2_2': tf.get_variable('conv2_2', shape=[3, 3, 128, 128],
                                   initializer=tf.contrib.layers.xavier_initializer()),
        'conv3_1': tf.get_variable('conv3_1', shape=[3, 3, 128, 256],
                                   initializer=tf.contrib.layers.xavier_initializer()),
        'conv3_2': tf.get_variable('conv3_2', shape=[3, 3, 256, 256],
                                   initializer=tf.contrib.layers.xavier_initializer()),
        'conv3_3': tf.get_variable('conv3_3', shape=[3, 3, 256, 256],
                                   initializer=tf.contrib.layers.xavier_initializer()),
        'conv3_4': tf.get_variable('conv3_4', shape=[3, 3, 256, 256],
                                   initializer=tf.contrib.layers.xavier_initializer()),
        'fc1': tf.get_variable('fc1', shape=[16384, 4096], initializer=tf.contrib.layers.xavier_initializer()),
        'defc1': tf.get_variable('defc1', shape=[4096, 16384], initializer=tf.contrib.layers.xavier_initializer()),
        'upsample3': tf.get_variable('upsample3', shape=[3, 3, 512, 512],
                                     initializer=tf.contrib.layers.xavier_initializer()),
        'deconv3_4': tf.get_variable('deconv3_4', shape=[3, 3, 64, 512],
                                     initializer=tf.contrib.layers.xavier_initializer()),
        'deconv3_3': tf.get_variable('deconv3_3', shape=[3, 3, 64, 64],
                                     initializer=tf.contrib.layers.xavier_initializer()),
        'deconv3_2': tf.get_variable('deconv3_2', shape=[3, 3, 64, 64],
                                     initializer=tf.contrib.layers.xavier_initializer()),
        'deconv3_1': tf.get_variable('deconv3_1', shape=[3, 3, 128, 64],
                                     initializer=tf.contrib.layers.xavier_initializer()),
        'upsample2': tf.get_variable('upsample2', shape=[3, 3, 256, 256],
                                     initializer=tf.contrib.layers.xavier_initializer()),

        'deconv2_2': tf.get_variable('deconv2_2', shape=[3, 3, 64, 256],
                                     initializer=tf.contrib.layers.xavier_initializer()),
        'deconv2_1': tf.get_variable('deconv2_1', shape=[3, 3, 64, 64],
                                     initializer=tf.contrib.layers.xavier_initializer()),
        'upsample1': tf.get_variable('upsample1', shape=[3, 3, 128, 128],
                                     initializer=tf.contrib.layers.xavier_initializer()),
        'deconv1_2': tf.get_variable('deconv1_2', shape=[3, 3, 64, 128],
                                     initializer=tf.contrib.layers.xavier_initializer()),
        'deconv1_1': tf.get_variable('deconv1_1', shape=[3, 3, 32, 64],
                                     initializer=tf.contrib.layers.xavier_initializer()),
        'upsample0': tf.get_variable('upsample0', shape=[3, 3, 64, 64],
                                     initializer=tf.contrib.layers.xavier_initializer()),
        'deconv0_2': tf.get_variable('deconv0_2', shape=[3, 3, 32, 64],
                                     initializer=tf.contrib.layers.xavier_initializer()),
        'deconv0_1': tf.get_variable('deconv0_1', shape=[3, 3, 32, 32],
                                     initializer=tf.contrib.layers.xavier_initializer()),
        'deconv': tf.get_variable('deconv', shape=[3, 3, 3, 32], initializer=tf.contrib.layers.xavier_initializer())
    }
with tf.variable_scope('biases'):
    bias = {
        'conv0_1': tf.get_variable('conv0_1', shape=[32], initializer=tf.constant_initializer(value=0)),
        'conv0_2': tf.get_variable('conv0_2', shape=[32], initializer=tf.constant_initializer(value=0)),
        'conv1_1': tf.get_variable('conv1_1', shape=[64], initializer=tf.constant_initializer(value=0)),
        'conv1_2': tf.get_variable('conv1_2', shape=[64], initializer=tf.constant_initializer(value=0)),
        'conv2_1': tf.get_variable('conv2_1', shape=[128], initializer=tf.constant_initializer(value=0)),
        'conv2_2': tf.get_variable('conv2_2', shape=[128], initializer=tf.constant_initializer(value=0)),
        'conv3_1': tf.get_variable('conv3_1', shape=[256], initializer=tf.constant_initializer(value=0)),
        'conv3_2': tf.get_variable('conv3_2', shape=[256], initializer=tf.constant_initializer(value=0)),
        'conv3_3': tf.get_variable('conv3_3', shape=[256], initializer=tf.constant_initializer(value=0)),
        'conv3_4': tf.get_variable('conv3_4', shape=[256], initializer=tf.constant_initializer(value=0)),
        'fc1': tf.get_variable('fc1', shape=[4096], initializer=tf.constant_initializer(value=0)),
        'defc1': tf.get_variable('defc1', shape=[16384], initializer=tf.constant_initializer(value=0)),
        'upsample3': tf.get_variable('upsample3', shape=[512],
                                     initializer=tf.constant_initializer(value=0)),
        'deconv3_4': tf.get_variable('deconv3_4', shape=[64],
                                     initializer=tf.constant_initializer(value=0)),
        'deconv3_3': tf.get_variable('deconv3_3', shape=[64],
                                     initializer=tf.constant_initializer(value=0)),
        'deconv3_2': tf.get_variable('deconv3_2', shape=[64],
                                     initializer=tf.constant_initializer(value=0)),
        'deconv3_1': tf.get_variable('deconv3_1', shape=[128],
                                     initializer=tf.constant_initializer(value=0)),
        'upsample2': tf.get_variable('upsample2', shape=[256],
                                     initializer=tf.constant_initializer(value=0)),

        'deconv2_2': tf.get_variable('deconv2_2', shape=[64],
                                     initializer=tf.constant_initializer(value=0)),
        'deconv2_1': tf.get_variable('deconv2_1', shape=[64],
                                     initializer=tf.constant_initializer(value=0)),
        'upsample1': tf.get_variable('upsample1', shape=[128],
                                     initializer=tf.constant_initializer(value=0)),
        'deconv1_2': tf.get_variable('deconv1_2', shape=[64],
                                     initializer=tf.constant_initializer(value=0)),
        'deconv1_1': tf.get_variable('deconv1_1', shape=[32],
                                     initializer=tf.constant_initializer(value=0)),
        'upsample0': tf.get_variable('upsample0', shape=[64],
                                     initializer=tf.constant_initializer(value=0)),
        'deconv0_2': tf.get_variable('deconv0_2', shape=[32],
                                     initializer=tf.constant_initializer(value=0)),
        'deconv0_1': tf.get_variable('deconv0_1', shape=[32],
                                     initializer=tf.constant_initializer(value=0)),
        'deconv': tf.get_variable('deconv', shape=[3],
                                  initializer=tf.constant_initializer(value=0))
    }

    def inference_G(image_batch,batch_size):
        conv0_1 = tf.nn.bias_add(tf.nn.conv2d(image_batch,weights['conv0_1'],strides=[1,1,1,1],padding='SAME'),bias['conv0_1'])
        conv0_1 = tf.nn.relu(conv0_1,name='relu0_1')

        conv0_2 = tf.nn.bias_add(tf.nn.conv2d(conv0_1,weights['conv0_2'],strides=[1,1,1,1],padding='SAME'),bias['conv0_2'])
        conv0_2 = tf.nn.relu(conv0_2,name='relu0_2')

        pool0 = tf.nn.max_pool(conv0_2,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')

        conv1_1 = tf.nn.bias_add(tf.nn.conv2d(pool0,weights['conv1_1'],strides=[1,1,1,1],padding='SAME'),bias['conv1_1'])
        conv1_1 = tf.nn.relu(conv1_1,name='relu1_1')

        conv1_2 = tf.nn.bias_add(tf.nn.conv2d(conv1_1,weights['conv1_2'],strides=[1,1,1,1],padding='SAME'),bias['conv1_2'])
        conv1_2 = tf.nn.relu(conv1_2,name='relu1_2')

        pool1 = tf.nn.max_pool(conv1_2,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')

        conv2_1 = tf.nn.bias_add(tf.nn.conv2d(pool1,weights['conv2_1'],strides=[1,1,1,1],padding='SAME'),bias['conv2_1'])
        conv2_1 = tf.nn.relu(conv2_1,name='relu2_1')

        conv2_2 = tf.nn.bias_add(tf.nn.conv2d(conv2_1,weights['conv2_2'],strides=[1,1,1,1],padding='SAME'),bias['conv2_2'])
        conv2_2 = tf.nn.relu(conv2_2,name='relu2_2')

        pool2 = tf.nn.max_pool(conv2_2,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')

        conv3_1 = tf.nn.bias_add(tf.nn.conv2d(pool2,weights['conv3_1'],strides=[1,1,1,1],padding='SAME'),bias['conv3_1'])
        conv3_1 = tf.nn.relu(conv3_1,name='relu3_1')

        conv3_2 = tf.nn.bias_add(tf.nn.conv2d(conv3_1 ,weights['conv3_2'], strides=[1, 1, 1, 1], padding='SAME'),
                                 bias['conv3_2'])
        conv3_2 = tf.nn.relu(conv3_2, name='relu3_2')

        conv3_3 = tf.nn.bias_add(tf.nn.conv2d(conv3_2, weights['conv3_3'], strides=[1, 1, 1, 1], padding='SAME'),
                                 bias['conv3_3'])
        conv3_3 = tf.nn.relu(conv3_3, name='relu3_3')

        conv3_4 = tf.nn.bias_add(tf.nn.conv2d(conv3_3, weights['conv3_4'], strides=[1, 1, 1, 1], padding='SAME'),
                                 bias['conv3_4'])
        conv3_4 = tf.nn.relu(conv3_4, name='relu3_4')

        pool3 = tf.nn.max_pool(conv3_4,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')

        reshape = pool3.get_shape().as_list()
        reshape_size = reshape[1]*reshape[2]*reshape[3]

        img_reshape = tf.reshape(pool3,[-1,reshape_size])

        fc1 = tf.matmul(img_reshape,weights['fc1'])+bias['fc1']

        defc1 = tf.matmul(fc1,weights['defc1'])+bias['defc1']

        img_reshape_2 = tf.reshape(defc1,[-1,8,8,256])

        img = tf.concat([pool3,img_reshape_2],axis=3)

        up3 = tf.nn.bias_add(tf.nn.conv2d_transpose(img,weights['upsample3'],output_shape=[batch_size,16,16,512],strides=[1,2,2,1],padding='SAME'),bias['upsample3'])

        deconv3_4 = tf.nn.bias_add(tf.nn.conv2d_transpose(up3,weights['deconv3_4'],output_shape=[batch_size,16,16,64],strides=[1,1,1,1],padding='SAME'),bias['deconv3_4'])
        deconv3_4 = tf.nn.relu(deconv3_4,name='derelu3_4')

        deconv3_3 = tf.nn.bias_add(tf.nn.conv2d_transpose(deconv3_4,weights['deconv3_3'],output_shape=[batch_size,16,16,64],strides=[1,1,1,1],padding='SAME'),bias['deconv3_3'])
        deconv3_3 = tf.nn.relu(deconv3_3,name='derelu3_3')

        deconv3_2 = tf.nn.bias_add(
            tf.nn.conv2d_transpose(deconv3_3, weights['deconv3_2'], output_shape=[batch_size, 16, 16, 64],
                                   strides=[1, 1, 1, 1], padding='SAME'), bias['deconv3_2'])
        deconv3_2 = tf.nn.relu(deconv3_2, name='derelu3_3')

        deconv3_1 = tf.nn.bias_add(
            tf.nn.conv2d_transpose(deconv3_2, weights['deconv3_1'], output_shape=[batch_size, 16, 16, 128],
                                   strides=[1, 1, 1, 1], padding='SAME'), bias['deconv3_1'])
        deconv3_1 = tf.nn.relu(deconv3_1, name='derelu3_1')
        img_2 = tf.concat([pool2,deconv3_1],axis=3)
        up2 = tf.nn.bias_add(tf.nn.conv2d_transpose(img_2,weights['upsample2'],output_shape=[batch_size,32,32,256],strides=[1,2,2,1],padding='SAME'),bias['upsample2'])

        deconv2_2 = tf.nn.bias_add(tf.nn.conv2d_transpose(up2,weights['deconv2_2'],output_shape=[batch_size,32,32,64],strides=[1,1,1,1],padding='SAME'),bias['deconv2_2'])
        deconv2_2 = tf.nn.relu(deconv2_2,name='derelu2_2')

        deconv2_1 = tf.nn.bias_add(tf.nn.conv2d_transpose(deconv2_2,weights['deconv2_1'],output_shape=[batch_size,32,32,64],strides=[1,1,1,1],padding='SAME'),bias['deconv2_1'])
        deconv2_1 = tf.nn.relu(deconv2_1,name='derelu2_1')

        img_3 = tf.concat([pool1,deconv2_1],axis=3)
        up1 = tf.nn.bias_add(tf.nn.conv2d_transpose(img_3,weights['upsample1'],output_shape=[batch_size,64,64,128],strides=[1,2,2,1],padding='SAME'),bias['upsample1'])

        deconv1_2 = tf.nn.bias_add(tf.nn.conv2d_transpose(up1,weights['deconv1_2'],output_shape=[batch_size,64,64,64],strides=[1,1,1,1],padding='SAME'),bias['deconv1_2'])
        deconv1_2 = tf.nn.relu(deconv1_2,name='derelu1_2')

        deconv1_1 = tf.nn.bias_add(tf.nn.conv2d_transpose(deconv1_2,weights['deconv1_1'],output_shape=[batch_size,64,64,32],strides=[1,1,1,1],padding='SAME'),bias['deconv1_1'])
        deconv1_1 = tf.nn.relu(deconv1_1,name='derelu1_1')

        img_3 = tf.concat([pool0,deconv1_1],axis=3)
        up0 = tf.nn.bias_add(tf.nn.conv2d_transpose(img_3,weights['upsample0'],output_shape=[batch_size,128,128,64],strides=[1,2,2,1],padding='SAME'),bias['upsample0'])

        deconv0_2 = tf.nn.bias_add(tf.nn.conv2d_transpose(up0,weights['deconv0_2'],output_shape=[batch_size,128,128,32],strides=[1,1,1,1],padding='SAME'),bias['deconv0_2'])
        deconv0_2 = tf.nn.relu(deconv0_2,name='derelu0_2')

        deconv0_1 = tf.nn.bias_add(tf.nn.conv2d_transpose(deconv0_2, weights['deconv0_1'],output_shape=[batch_size,128,128,32],strides=[1,1,1,1],padding='SAME'),bias['deconv0_1'])
        deconv0_1 = tf.nn.relu(deconv0_1,name='derelu0_1')

        deconv = tf.nn.bias_add(tf.nn.conv2d_transpose(deconv0_1,weights['deconv'],output_shape=[batch_size,128,128,3],strides=[1,1,1,1],padding='SAME'),bias['deconv'])

        return deconv











