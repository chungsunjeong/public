def form_image_grid(input_tensor, grid_shape, image_shape, num_channels):
    if grid_shape[0] * grid_shape[1] != int(input_tensor.get_shape()[0]):
        raise ValueError('Grid shape incompatible with minibatch size.')
    if len(input_tensor.get_shape()) == 2:
        num_features = image_shape[0] * image_shape[1] * num_channels
        if int(input_tensor.get_shape()[1]) != num_features:
            raise ValueError('Image shape and number of channels incompatible with '
                             'input tensor.')
    elif len(input_tensor.get_shape()) == 4:
        if (int(input_tensor.get_shape()[1]) != image_shape[0] or
                int(input_tensor.get_shape()[2]) != image_shape[1] or
                int(input_tensor.get_shape()[3]) != num_channels):
            raise ValueError('Image shape and number of channels incompatible with '
                             'input tensor.')
    else:
        raise ValueError('Unrecognized input tensor format.')
    height, width = grid_shape[0] * image_shape[0], grid_shape[1] * image_shape[1]
    input_tensor = tf.reshape(
        input_tensor, grid_shape + image_shape + [num_channels])
    input_tensor = tf.transpose(input_tensor, [0, 1, 3, 2, 4])
    input_tensor = tf.reshape(
        input_tensor, [grid_shape[0], width, image_shape[0], num_channels])
    input_tensor = tf.transpose(input_tensor, [0, 2, 1, 3])
    input_tensor = tf.reshape(
        input_tensor, [1, height, width, num_channels])
    return input_tensor