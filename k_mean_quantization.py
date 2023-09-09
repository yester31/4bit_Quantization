from base_model import *
from collections import namedtuple
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap

import copy
import torch
from torch import nn
from fast_pytorch_kmeans import KMeans

Codebook = namedtuple('Codebook', ['centroids', 'labels'])

def k_means_quantize(fp32_tensor: torch.Tensor, bitwidth=4, codebook=None):
    """
    quantize tensor using k-means clustering
    :param fp32_tensor:
    :param bitwidth: [int] quantization bit width, default=4
    :param codebook: [Codebook] (the cluster centroids, the cluster label tensor)
    :return:
        [Codebook = (centroids, labels)]
            centroids: [torch.(cuda.)FloatTensor] the cluster centroids
            labels: [torch.(cuda.)LongTensor] cluster label tensor
    """
    if codebook is None:
        ############### YOUR CODE STARTS HERE ###############
        # get number of clusters based on the quantization precision
        # hint: one line of code
        n_clusters = 1 << bitwidth
        ############### YOUR CODE ENDS HERE #################
        # use k-means to get the quantization centroids
        kmeans = KMeans(n_clusters=n_clusters, mode='euclidean', verbose=0)
        labels = kmeans.fit_predict(fp32_tensor.view(-1, 1)).to(torch.long)
        centroids = kmeans.centroids.to(torch.float).view(-1)
        codebook = Codebook(centroids, labels)
    ############### YOUR CODE STARTS HERE ###############
    # decode the codebook into k-means quantized tensor for inference
    # hint: one line of code
    quantized_tensor = codebook.centroids[codebook.labels]
    ############### YOUR CODE ENDS HERE #################
    fp32_tensor.set_(quantized_tensor.view_as(fp32_tensor))
    return codebook

def test_k_means_quantize(
    test_tensor=torch.tensor([
        [-0.3747,  0.0874,  0.3200, -0.4868,  0.4404],
        [-0.0402,  0.2322, -0.2024, -0.4986,  0.1814],
        [ 0.3102, -0.3942, -0.2030,  0.0883, -0.4741],
        [-0.1592, -0.0777, -0.3946, -0.2128,  0.2675],
        [ 0.0611, -0.1933, -0.4350,  0.2928, -0.1087]]),
    bitwidth=2):
    def plot_matrix(tensor, ax, title, cmap=ListedColormap(['white'])):
        ax.imshow(tensor.cpu().numpy(), vmin=-0.5, vmax=0.5, cmap=cmap)
        ax.set_title(title)
        ax.set_yticklabels([])
        ax.set_xticklabels([])
        for i in range(tensor.shape[1]):
            for j in range(tensor.shape[0]):
                text = ax.text(j, i, f'{tensor[i, j].item():.2f}',
                                ha="center", va="center", color="k")

    fig, axes = plt.subplots(1,2, figsize=(8, 12))
    ax_left, ax_right = axes.ravel()

    plot_matrix(test_tensor, ax_left, 'original tensor')

    num_unique_values_before_quantization = test_tensor.unique().numel()
    k_means_quantize(test_tensor, bitwidth=bitwidth)
    num_unique_values_after_quantization = test_tensor.unique().numel()
    print('* Test k_means_quantize()')
    print(f'    target bitwidth: {bitwidth} bits')
    print(f'        num unique values before k-means quantization: {num_unique_values_before_quantization}')
    print(f'        num unique values after  k-means quantization: {num_unique_values_after_quantization}')
    assert num_unique_values_after_quantization == min((1 << bitwidth), num_unique_values_before_quantization)
    print('* Test passed.')

    plot_matrix(test_tensor, ax_right, f'{bitwidth}-bit k-means quantized tensor', cmap='tab20c')
    fig.tight_layout()
    plt.show()

from torch.nn import parameter
def update_codebook(fp32_tensor: torch.Tensor, codebook: Codebook):
    """
    update the centroids in the codebook using updated fp32_tensor
    :param fp32_tensor: [torch.(cuda.)Tensor]
    :param codebook: [Codebook] (the cluster centroids, the cluster label tensor)
    """
    n_clusters = codebook.centroids.numel()
    fp32_tensor = fp32_tensor.view(-1)
    for k in range(n_clusters):
    ############### YOUR CODE STARTS HERE ###############
        # hint: one line of code
        codebook.centroids[k] = fp32_tensor[codebook.labels == k].mean()
    ############### YOUR CODE ENDS HERE #################

class KMeansQuantizer:
    def __init__(self, model : nn.Module, bitwidth=4):
        self.codebook = KMeansQuantizer.quantize(model, bitwidth)

    @torch.no_grad()
    def apply(self, model, update_centroids):
        for name, param in model.named_parameters():
            if name in self.codebook:
                if update_centroids:
                    update_codebook(param, codebook=self.codebook[name])
                self.codebook[name] = k_means_quantize(
                    param, codebook=self.codebook[name])

    @staticmethod
    @torch.no_grad()
    def quantize(model: nn.Module, bitwidth=4):
        codebook = dict()
        if isinstance(bitwidth, dict):
            for name, param in model.named_parameters():
                if name in bitwidth:
                    codebook[name] = k_means_quantize(param, bitwidth=bitwidth[name])
        else:
            for name, param in model.named_parameters():
                if param.dim() > 1:
                    codebook[name] = k_means_quantize(param, bitwidth=bitwidth)
        return codebook

def update_codebook(fp32_tensor: torch.Tensor, codebook: Codebook):
    """
    update the centroids in the codebook using updated fp32_tensor
    :param fp32_tensor: [torch.(cuda.)Tensor]
    :param codebook: [Codebook] (the cluster centroids, the cluster label tensor)
    """
    n_clusters = codebook.centroids.numel()
    fp32_tensor = fp32_tensor.view(-1)
    for k in range(n_clusters):
    ############### YOUR CODE STARTS HERE ###############
        # hint: one line of code
        codebook.centroids[k] = fp32_tensor[codebook.labels == k].mean()
    ############### YOUR CODE ENDS HERE #################

if __name__ == "__main__":
    # test_k_means_quantize()

    model_name = 'vgg'
    model = VGG().cuda()
    checkpoint_path = f"./checkpoints/best_{model_name}.pth.tar"
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        print(f"=> loading checkpoint '{checkpoint_path}'")
        model.load_state_dict(checkpoint)
    else:
        train_flag = True

    recover_model = lambda: model.load_state_dict(checkpoint)

    transforms = {
        "train": Compose([
            RandomCrop(32, padding=4),
            RandomHorizontalFlip(),
            ToTensor(),
        ]),
        "test": ToTensor(),
    }
    dataset = {}
    for split in ["train", "test"]:
        dataset[split] = CIFAR10(
            root="data/cifar10",
            train=(split == "train"),
            download=True,
            transform=transforms[split],
        )
    dataloader = {}
    for split in ['train', 'test']:
        dataloader[split] = DataLoader(
            dataset[split],
            batch_size=512,
            shuffle=(split == 'train'),
            num_workers=0,
            pin_memory=True,
        )

    fp32_model_accuracy = evaluate(model, dataloader['test'])
    fp32_model_size = get_model_size(model)
    print(f"fp32 model has accuracy={fp32_model_accuracy:.2f}%")
    print(f"fp32 model has size={fp32_model_size / MiB:.2f} MiB")

    use_amp = True
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    print('Note that the storage for codebooks is ignored when calculating the model size.')
    quantizers = dict()
    for bitwidth in [8, 4, 2]:
        recover_model()
        print(f'k-means quantizing model into {bitwidth} bits')
        quantizer = KMeansQuantizer(model, bitwidth)
        quantized_model_size = get_model_size(model, bitwidth)
        print(f"    {bitwidth}-bit k-means quantized model has size={quantized_model_size / MiB:.2f} MiB")
        quantized_model_accuracy = evaluate(model, dataloader['test'])
        print(f"    {bitwidth}-bit k-means quantized model has accuracy={quantized_model_accuracy:.2f}%")
        quantizers[bitwidth] = quantizer

    accuracy_drop_threshold = 0.5
    quantizers_before_finetune = copy.deepcopy(quantizers)
    quantizers_after_finetune = quantizers


    for bitwidth in [8, 4, 2]:
        recover_model()
        quantizer = quantizers[bitwidth]
        print(f'k-means quantizing model into {bitwidth} bits')
        quantizer.apply(model, update_centroids=False)
        quantized_model_size = get_model_size(model, bitwidth)
        print(f"    {bitwidth}-bit k-means quantized model has size={quantized_model_size / MiB:.2f} MiB")
        quantized_model_accuracy = evaluate(model, dataloader['test'])
        print(
            f"    {bitwidth}-bit k-means quantized model has accuracy={quantized_model_accuracy:.2f}% before quantization-aware training ")
        accuracy_drop = fp32_model_accuracy - quantized_model_accuracy
        if accuracy_drop > accuracy_drop_threshold:
            print(
                f"        Quantization-aware training due to accuracy drop={accuracy_drop:.2f}% is larger than threshold={accuracy_drop_threshold:.2f}%")
            num_finetune_epochs = 5
            optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, num_finetune_epochs)
            criterion = nn.CrossEntropyLoss()
            best_accuracy = 0
            epoch = num_finetune_epochs
            while accuracy_drop > accuracy_drop_threshold and epoch > 0:
                train(model, dataloader['train'], criterion, optimizer, scheduler, scaler, use_amp,
                      callbacks=[lambda: quantizer.apply(model, update_centroids=True)])
                model_accuracy = evaluate(model, dataloader['test'])
                is_best = model_accuracy > best_accuracy
                best_accuracy = max(model_accuracy, best_accuracy)
                print(
                    f'        Epoch {num_finetune_epochs - epoch} Accuracy {model_accuracy:.2f}% / Best Accuracy: {best_accuracy:.2f}%')
                accuracy_drop = fp32_model_accuracy - best_accuracy
                epoch -= 1
        else:
            print(
                f"        No need for quantization-aware training since accuracy drop={accuracy_drop:.2f}% is smaller than threshold={accuracy_drop_threshold:.2f}%")