import torch
import json
import numpy as np
import numpy.typing as npt
from lib.misc import datasets_root_path, project_root_path
import lib.open_fwi.network as network
from lib.visualize import show_velocity_model
import lib.open_fwi.transforms as T
from torchvision.transforms import Compose
import matplotlib.pyplot as plt


def predict(seismic_data_batch, dataset_config: dict):
    # modelのrestore
    model = network.model_dict['InversionNet'](
        upsample_mode=None,
        sample_spatial=1.0,
        sample_temporal=1,
        norm='bn'
    ).to('cpu')
    # checkpoint = torch.load(datasets_root_path.joinpath('open-fwi/tmp/fva_l2_VelocityGAN.pth'), map_location='cpu')
    checkpoint = torch.load(datasets_root_path.joinpath('open-fwi/tmp/fva_l2_InversionNet.pth'), map_location='cpu')
    model.load_state_dict(network.replace_legacy(checkpoint['model']))
    model.eval()

    # 入力データの変換(変換後のデータを用いて学習されている）
    # 1. 変換器の作成
    k = 1
    log_data_min = T.log_transform(dataset_config['data_min'], k=k)
    log_data_max = T.log_transform(dataset_config['data_max'], k=k)
    transform = Compose([
        T.LogTransform(k=k),
        T.MinMaxNormalize(log_data_min, log_data_max),
    ])
    # 2. 変換
    transformed_seismic_data_batch = transform(seismic_data_batch)

    # 3. モデルに入力するための変換(numpy -> torch)
    torch_train_data = torch.from_numpy(transformed_seismic_data_batch.astype(np.float32)).clone()

    with torch.no_grad():
        torch_pred = model(torch_train_data)
    pred = torch_pred.cpu().numpy()
    return pred


def shift_vertical(signal: npt.NDArray, shift: int):
    if shift >= 0:
        return np.vstack((np.tile(signal[0], (shift, 1)), signal))[:-shift]
    else:
        return np.vstack((signal, np.tile(signal[-1], (-shift, 1))))[-shift:]


def main():
    target_idx = 0

    with open(project_root_path.joinpath('lib/open_fwi/dataset_config.json')) as f:
        dataset_config = json.load(f)['flatvel-a']

    true_train_data = np.load(datasets_root_path.joinpath('open-fwi/tmp/data1.npy'))[target_idx:target_idx+1]
    raw_true_data = np.load(datasets_root_path.joinpath('open-fwi/tmp/model1.npy'))[target_idx:target_idx+1]

    train_data = np.zeros((1, 5, 1000, 70))
    noise_sigma = 0.1
    noise = np.random.normal(0, noise_sigma, train_data.shape)
    # train_data[0] = np.load(datasets_root_path.joinpath('open-fwi/tmp/true_observed_waveforms.npy'))
    # train_data[0] = true_train_data[0]
    train_data[0] = true_train_data[0] + noise
    # train_data[0] = noise

    print(train_data.shape)

    # for i in range(5):
    #     train_data[0][i] = shift_vertical(train_data[0][i], 6)

    print("train data RMSE: ", np.sqrt(np.mean((train_data[0] - true_train_data[target_idx]) ** 2)), ", ", np.sum(np.abs(train_data[0] - true_train_data[target_idx])))


    # plt.imshow(true_train_data[0][2], extent=[0, 1, 0, 1]), plt.title('original train seismic data'), plt.colorbar(), plt.axis('off'), plt.show()
    # plt.imshow(true_train_data[0][2] + noise[0][2], extent=[0, 1, 0, 1]), plt.title('original train seismic data + noise(σ=1)'), plt.colorbar(), plt.axis('off'), plt.show()
    plt.imshow(train_data[0][2], extent=[0, 1, 0, 1]), plt.title('seismic data with devito simulation'), plt.colorbar(), plt.axis('off'), plt.show()


    transform_valid_label = Compose([
        T.MinMaxNormalize(dataset_config['label_min'], dataset_config['label_max'])
    ])
    true_data = transform_valid_label(raw_true_data.copy())

    restore = lambda x: ((x / 2 + 0.5) * (dataset_config['label_max'] - dataset_config['label_min']) + dataset_config['label_min']) / 1000.

    pred = predict(train_data, dataset_config=dataset_config)
    np.save(datasets_root_path.joinpath('open-fwi/tmp/pred.npy'), pred)

    show_velocity_model(raw_true_data[0][0] / 1000., title='True Velocity Model', vmin=1.5, vmax=4.5)
    show_velocity_model(restore(pred[0][0]), title='Predicted Velocity Model', vmin=1.5, vmax=4.5)
    print("RMSE: ", np.sqrt(np.mean((true_data[0][0] - pred[0][0]) ** 2)))


if __name__ == '__main__':
    main()
