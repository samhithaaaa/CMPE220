{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "metadata": {},
   "outputs": [],
   "source": [
    "import pickle\n",
    "import numpy as np\n",
    "import matplotlib.pyplot as plt\n",
    "from sklearn.preprocessing import LabelBinarizer\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "metadata": {},
   "outputs": [],
   "source": [
    "def _load_label_names():\n",
    "    \"\"\"\n",
    "    Load the label names from file\n",
    "    \"\"\"\n",
    "    return ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "metadata": {},
   "outputs": [],
   "source": [
    "def load_cfar10_batch(cifar10_dataset_folder_path, batch_id):\n",
    "    \"\"\"\n",
    "    Load a batch of the dataset\n",
    "    \"\"\"\n",
    "    with open(cifar10_dataset_folder_path + '/data_batch_' + str(batch_id), mode='rb') as file:\n",
    "        batch = pickle.load(file, encoding='latin1')\n",
    "\n",
    "    features = batch['data'].reshape((len(batch['data']), 3, 32, 32)).transpose(0, 2, 3, 1)\n",
    "    labels = batch['labels']\n",
    "\n",
    "    return features, labels"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "metadata": {},
   "outputs": [],
   "source": [
    "def display_stats(cifar10_dataset_folder_path, batch_id, sample_id):\n",
    "    \"\"\"\n",
    "    Display Stats of the the dataset\n",
    "    \"\"\"\n",
    "    batch_ids = list(range(1, 6))\n",
    "\n",
    "    if batch_id not in batch_ids:\n",
    "        print('Batch Id out of Range. Possible Batch Ids: {}'.format(batch_ids))\n",
    "        return None\n",
    "\n",
    "    features, labels = load_cfar10_batch(cifar10_dataset_folder_path, batch_id)\n",
    "\n",
    "    if not (0 <= sample_id < len(features)):\n",
    "        print('{} samples in batch {}.  {} is out of range.'.format(len(features), batch_id, sample_id))\n",
    "        return None\n",
    "\n",
    "    print('\\nStats of batch {}:'.format(batch_id))\n",
    "    print('Samples: {}'.format(len(features)))\n",
    "    print('Label Counts: {}'.format(dict(zip(*np.unique(labels, return_counts=True)))))\n",
    "    print('First 20 Labels: {}'.format(labels[:20]))\n",
    "\n",
    "    sample_image = features[sample_id]\n",
    "    sample_label = labels[sample_id]\n",
    "    label_names = _load_label_names()\n",
    "\n",
    "    print('\\nExample of Image {}:'.format(sample_id))\n",
    "    print('Image - Min Value: {} Max Value: {}'.format(sample_image.min(), sample_image.max()))\n",
    "    print('Image - Shape: {}'.format(sample_image.shape))\n",
    "    print('Label - Label Id: {} Name: {}'.format(sample_label, label_names[sample_label]))\n",
    "    plt.axis('off')\n",
    "    plt.imshow(sample_image)\n",
    "    plt.ion()\n",
    "    plt.show()\n",
    "    plt.pause(0.001)\n",
    "    input(\"Press [enter] to continue.\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "metadata": {},
   "outputs": [],
   "source": [
    "def _preprocess_and_save(normalize, one_hot_encode, features, labels, filename):\n",
    "    \"\"\"\n",
    "    Preprocess data and save it to file\n",
    "    \"\"\"\n",
    "    features = normalize(features)\n",
    "    labels = one_hot_encode(labels)\n",
    "\n",
    "    pickle.dump((features, labels), open(filename, 'wb'))\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "metadata": {},
   "outputs": [],
   "source": [
    "def preprocess_and_save_data(cifar10_dataset_folder_path, normalize, one_hot_encode):\n",
    "    \"\"\"\n",
    "    Preprocess Training and Validation Data\n",
    "    \"\"\"\n",
    "    n_batches = 5\n",
    "    valid_features = []\n",
    "    valid_labels = []\n",
    "\n",
    "    for batch_i in range(1, n_batches + 1):\n",
    "        features, labels = load_cfar10_batch(cifar10_dataset_folder_path, batch_i)\n",
    "        validation_count = int(len(features) * 0.1)\n",
    "\n",
    "        # Prprocess and save a batch of training data\n",
    "        _preprocess_and_save(\n",
    "            normalize,\n",
    "            one_hot_encode,\n",
    "            features[:-validation_count],\n",
    "            labels[:-validation_count],\n",
    "            'preprocess_batch_' + str(batch_i) + '.p')\n",
    "\n",
    "        # Use a portion of training batch for validation\n",
    "        valid_features.extend(features[-validation_count:])\n",
    "        valid_labels.extend(labels[-validation_count:])\n",
    "\n",
    "    # Preprocess and Save all validation data\n",
    "    _preprocess_and_save(\n",
    "        normalize,\n",
    "        one_hot_encode,\n",
    "        np.array(valid_features),\n",
    "        np.array(valid_labels),\n",
    "        'preprocess_validation.p')\n",
    "\n",
    "    with open(cifar10_dataset_folder_path + '/test_batch', mode='rb') as file:\n",
    "        batch = pickle.load(file, encoding='latin1')\n",
    "\n",
    "    # load the training data\n",
    "    test_features = batch['data'].reshape((len(batch['data']), 3, 32, 32)).transpose(0, 2, 3, 1)\n",
    "    test_labels = batch['labels']\n",
    "\n",
    "    # Preprocess and Save all training data\n",
    "    _preprocess_and_save(\n",
    "        normalize,\n",
    "        one_hot_encode,\n",
    "        np.array(test_features),\n",
    "        np.array(test_labels),\n",
    "        'preprocess_training.p')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "metadata": {},
   "outputs": [],
   "source": [
    "def batch_features_labels(features, labels, batch_size):\n",
    "    \"\"\"\n",
    "    Split features and labels into batches\n",
    "    \"\"\"\n",
    "    for start in range(0, len(features), batch_size):\n",
    "        end = min(start + batch_size, len(features))\n",
    "        yield features[start:end], labels[start:end]\n",
    "\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "metadata": {},
   "outputs": [],
   "source": [
    "def load_preprocess_training_batch(batch_id, batch_size):\n",
    "    \"\"\"\n",
    "    Load the Preprocessed Training data and return them in batches of <batch_size> or less\n",
    "    \"\"\"\n",
    "    filename = 'preprocess_batch_' + str(batch_id) + '.p'\n",
    "    features, labels = pickle.load(open(filename, mode='rb'))\n",
    "\n",
    "    # Return the training data in batches of size <batch_size> or less\n",
    "    return batch_features_labels(features, labels, batch_size)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "metadata": {},
   "outputs": [],
   "source": [
    "def load_preprocess_training_batch(batch_id, batch_size):\n",
    "    \"\"\"\n",
    "    Load the Preprocessed Training data and return them in batches of <batch_size> or less\n",
    "    \"\"\"\n",
    "    filename = 'preprocess_batch_' + str(batch_id) + '.p'\n",
    "    features, labels = pickle.load(open(filename, mode='rb'))\n",
    "\n",
    "    # Return the training data in batches of size <batch_size> or less\n",
    "    return batch_features_labels(features, labels, batch_size)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "metadata": {},
   "outputs": [],
   "source": [
    "def display_image_predictions(features, labels, predictions):\n",
    "    n_classes = 10\n",
    "    label_names = _load_label_names()\n",
    "    label_binarizer = LabelBinarizer()\n",
    "    label_binarizer.fit(range(n_classes))\n",
    "    label_ids = label_binarizer.inverse_transform(np.array(labels))\n",
    "\n",
    "    fig, axies = plt.subplots(nrows=4, ncols=2)\n",
    "    fig.tight_layout()\n",
    "    fig.suptitle('Softmax Predictions', fontsize=20, y=1.1)\n",
    "\n",
    "    n_predictions = 3\n",
    "    margin = 0.05\n",
    "    ind = np.arange(n_predictions)\n",
    "    width = (1. - 2. * margin) / n_predictions\n",
    "\n",
    "    for image_i, (feature, label_id, pred_indicies, pred_values) in enumerate(zip(features, label_ids, predictions.indices, predictions.values)):\n",
    "        pred_names = [label_names[pred_i] for pred_i in pred_indicies]\n",
    "        correct_name = label_names[label_id]\n",
    "\n",
    "        axies[image_i][0].imshow(feature)\n",
    "        axies[image_i][0].set_title(correct_name)\n",
    "        axies[image_i][0].set_axis_off()\n",
    "\n",
    "        axies[image_i][1].barh(ind + margin, pred_values[::-1], width)\n",
    "        axies[image_i][1].set_yticks(ind + margin)\n",
    "        axies[image_i][1].set_yticklabels(pred_names[::-1])\n",
    "        axies[image_i][1].set_xticks([0, 0.5, 1.0])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "metadata": {},
   "outputs": [
    {
     "ename": "NameError",
     "evalue": "name 'tests' is not defined",
     "output_type": "error",
     "traceback": [
      "\u001b[1;31m---------------------------------------------------------------------------\u001b[0m",
      "\u001b[1;31mNameError\u001b[0m                                 Traceback (most recent call last)",
      "\u001b[1;32m<ipython-input-12-fd738ac4cc9f>\u001b[0m in \u001b[0;36m<module>\u001b[1;34m\u001b[0m\n\u001b[0;32m      6\u001b[0m \u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m      7\u001b[0m \u001b[1;33m\u001b[0m\u001b[0m\n\u001b[1;32m----> 8\u001b[1;33m \u001b[0mtests\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mtest_one_hot_encode\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mone_hot_encode\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0m",
      "\u001b[1;31mNameError\u001b[0m: name 'tests' is not defined"
     ]
    }
   ],
   "source": [
    "def one_hot_encode(x):\n",
    "    classes = list(range(10))\n",
    "    lb = preprocessing.LabelBinarizer()\n",
    "    lb.fit(classes)\n",
    "    return lb.transform(x)\n",
    "   \n",
    "    \n",
    "tests.test_one_hot_encode(one_hot_encode)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.6.6"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}
