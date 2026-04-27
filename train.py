import torch
import torch.nn as nn
import torch.optim as optim
import os
from opts.train_opt import TrainOpt
from data.dataloader import CustomDatasetDataLoader
from utils import utils
from networks.networks import NetworksFactory
from tqdm import tqdm
from torch.optim.lr_scheduler import StepLR


if __name__ == '__main__':
    opt = TrainOpt().parse()
    # 设置设备 (GPU or CPU)
    ''' 使用CPU或CUDA '''
    device = torch.device(opt.cuda)
    if 'cuda' in opt.cuda:
        torch.cuda.set_device(device)


    # 加载数据集
    data_loader_train = CustomDatasetDataLoader(opt, is_for_train=True)
    dataset_train = data_loader_train.load_data()
    dataset_train_size = len(data_loader_train)
    print('# Training images : %d' % dataset_train_size)

    # 初始化网络
    model = NetworksFactory.get_by_name('our')
    # Load trained model
    initial_epoch = utils.findLastCheckpoint(save_dir="Checkpoint")  # load the last model in matconvnet style
    if initial_epoch > 0:
        print('Load model: resuming by loading epoch %03d' % initial_epoch)
        model = torch.load(os.path.join("Checkpoint", 'model_%03d.pth' % initial_epoch))
    model = model.to(device)

    loss_fn_l1 = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=opt.learning_rate)
    scheduler = StepLR(optimizer, step_size=20, gamma=0.8)

    checkpoint_dir = "Checkpoint"
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    for epoch in range(initial_epoch, opt.num_epochs):
        model.train()
        running_loss = 0.0
        with tqdm(dataset_train, unit="batch") as tepoch:
            for i, train_batch in enumerate (tepoch):

                blurred = train_batch['blurred'].to(device)
                event_bins = train_batch['event_bins'].to(device)
                gt = train_batch['gt'].to(device)

                outputs, L, loss_consistency = model(blurred, event_bins)

                loss = loss_fn_l1(outputs, gt) + loss_fn_l1(L, gt) + 0.4 * loss_consistency


                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
                tepoch.set_postfix(loss=loss.item())

        scheduler.step()
        print(f"Epoch [{epoch + 1}/{opt.num_epochs}], Loss: {running_loss / len(dataset_train):.4f}")


        if (epoch + 1) % 10 == 0:  # 保存每10个epoch的训练模型
            model_path = os.path.join(checkpoint_dir, 'model_%03d.pth' % (epoch + 1))
            torch.save(model, model_path)
            print('Saving checkpoint model_%03d.pth' % (epoch + 1))

    print('Finished Training')