import os
import pathlib
from project.ganCollections import *
import cs236781.plot as plot
import cs236781.download
import torchvision.transforms as T
from torchvision.datasets import ImageFolder
import IPython.display
import tqdm
from scipy.interpolate import UnivariateSpline
from IPython.display import Image, display
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure



DATA_DIR = pathlib.Path().absolute().joinpath('project/pytorch-datasets')
DEFAULT_DATA_URL = 'http://vis-www.cs.umass.edu/lfw/lfw-a.zip'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

image_size = 64
step = 10

def generate_data_set(data_url=DEFAULT_DATA_URL):
    _, dataset_dir = cs236781.download.download_data(out_path=DATA_DIR, url=data_url, extract=True, force=False)

    tf = T.Compose([
        # Resize to constant spatial dimensions
        T.Resize((image_size, image_size)),
        # PIL.Image -> torch.Tensor
        T.ToTensor(),
        # Dynamic range [0,1] -> [-1, 1]
        T.Normalize(mean=(.5, .5, .5), std=(.5, .5, .5)),
    ])

    ds_gwb = ImageFolder(os.path.dirname(dataset_dir), tf)
    return ds_gwb

def generate_results(train=False, num_epochs=10):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    gan_model_names = ['Vanila-Base-Gan', 'SN-Gan', 'W-Gan', 'SN-W-Gan']
    results_dir_path = 'project/final-results/'
    if train:
        train_models(gan_model_names, num_epochs)
        results_dir_path = 'project/results/'
    print('Generating Results')
    for name in gan_model_names:
            model_path = results_dir_path + name
            generate_results_from_path(name, model_path)
    print('Inception Score Comparition:')
    plot_score(results_dir_path, gan_model_names, num_epochs)


def plot_score(path, ganModelNames, num_epochs):

    x = np.arange(0, num_epochs, step)
    xs = np.linspace(0, x[-1] - 1, 100)
    c = ['k', 'r', 'b', 'g']
    legends = []
    figure(figsize=(14, 8))
    for idx, modelName in enumerate(ganModelNames):
        scores_file = path + modelName + '/scores.pt'
        y = torch.load(scores_file)
        y = [IS[0] for IS in y]
        x = [i*step for i in range(0,len(y))]
        xs = np.linspace(0, x[-1] - 1, 100)
        s = UnivariateSpline(x, y, s=5)
        ys = s(xs)
        plt.plot(x, y, f'{c[idx]}.')
        plt.plot(xs, ys, f'{c[idx]}--')
        legends += [f'{modelName} scores', f'{modelName} trend curve']
    plt.title("Inception Score During Training")
    plt.legend(legends, loc='lower right')
    plt.xlabel("Epoch")
    plt.ylabel("Inception Score ")
    plt.grid(axis='both')
    plt.show()

def train_models(gan_model_names, num_epochs=10):

    ds_gwb = generate_data_set()
    x0, y0 = ds_gwb[0]
    x0 = x0.unsqueeze(0).to(device)

    batch_size = DLparams['batch_size']
    im_size = ds_gwb[0][0].shape

    dl_train = DataLoader(ds_gwb, batch_size, shuffle=True)

    # constract new models
    gan_models = dict()
    if 'Vanila-Base-Gan' in gan_model_names:
        gan_models['Vanila-Base-Gan'] = Gan(im_size, device=device)
    if 'SN-Gan' in gan_model_names:
        gan_models['SN-Gan'] = SnGan(im_size, device=device)
    if 'W-Gan' in gan_model_names:
        gan_models['W-Gan'] = WGan(im_size, device=device)
    if 'SN-W-Gan' in gan_model_names:
        gan_models['SN-W-Gan'] = SnWGan(im_size, device=device)
    # train
    try:
        model_file_path = {}
        for gan_name, gan_module in gan_models.items():
            model_file_path[gan_name] = pathlib.Path().parent.absolute().joinpath(
                f'project/results/{gan_name}/')
            if not os.path.exists(model_file_path[gan_name]):
                os.makedirs(model_file_path[gan_name])

        scores, dsc_avg_losses, gen_avg_losses ={}, {}, {}
        for gan_name, gan_module in gan_models.items():
            dsc_avg_losses[gan_name] = []
            gen_avg_losses[gan_name] = []
            scores[gan_name] = []

        for epoch_idx in range(num_epochs):
            # We'll accumulate batch losses and show an average once per epoch.
            dsc_losses, gen_losses = {}, {}
            for gan_name, gan_module in gan_models.items():
                dsc_losses[gan_name] = []
                gen_losses[gan_name] = []
            print(f'--- EPOCH {epoch_idx + 1}/{num_epochs} ---')

            with tqdm.tqdm(total=len(dl_train.batch_sampler), file=sys.stdout) as pbar:
                for batch_idx, (x_data, _) in enumerate(dl_train):
                    x_data = x_data.to(device)
                    for gan_name, gan_module in gan_models.items():
                        dsc_loss, gen_loss = gan_module.train_batch(x_data)
                        dsc_losses[gan_name].append(dsc_loss)
                        gen_losses[gan_name].append(gen_loss)
                    pbar.update()
            for gan_name, gan_module in gan_models.items():
                gen = gan_module.generator
                dsc_avg_losses[gan_name].append(np.mean(dsc_losses[gan_name]))
                gen_avg_losses[gan_name].append(np.mean(gen_losses[gan_name]))
                if epoch_idx % step == 0:
                    print(f'{gan_name}')
                    print(f'Discriminator loss: {dsc_avg_losses[gan_name][-1]}')
                    print(f'Generator loss:     {gen_avg_losses[gan_name][-1]}')

            for gan_name, gan_module in gan_models.items():
                if epoch_idx % step == 0:
                    print(f'========{gan_name}========')
                    gen = gan_module.generator
                    samples = gen.sample(5, with_grad=False)
                    fig, _ = plot.tensors_as_images(samples.cpu(), figsize=(6, 2))
                    # torch.save(fig, model_file_path[gan_name].joinpath('fig.pt'))
                    IPython.display.display(fig)
                    plt.close(fig)
                    print(f'========================')
                    scores[gan_name] += [inception_score(gen, cuda=True, batch_size=32, resize=True, splits=10, len=20000)]
                    torch.save(scores[gan_name], model_file_path[gan_name].joinpath('scores.pt'))
                    print(f'Inception score for {gan_name} is: mean: {scores[gan_name][-1][0]} std: {scores[gan_name][-1][1]}' )


        gan_models = {gan_name: gan_module.generator for gan_name, gan_module in gan_models.items()}
    except KeyboardInterrupt as e:
        print('\n *** Training interrupted by user')

    for gan_name, gen in gan_models.items():
        samples = gen.sample(50, with_grad=False)
        torch.save(samples, model_file_path[gan_name].joinpath('generated_samples.pt'))
    print('Training Complete')



def generate_results_from_path(modelName, modelPath):
    print('=====================================')
    print(f'Model Type: {modelName}')
    print(f'Generated Images: ')
    samples = torch.load(f'{modelPath}/generated_samples.pt')
    fig, _ = plot.tensors_as_images(samples.cpu(), figsize=(15,10), nrows=5)
    IPython.display.display(fig)
    plt.close(fig)
    print('=====================================')