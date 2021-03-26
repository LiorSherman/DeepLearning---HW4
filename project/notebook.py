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
#http://vis-www.cs.umass.edu/lfw/lfw-a.zip
#http://vis-www.cs.umass.edu/lfw/lfw-bush.zip
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

image_size = 64
step = 10
Unlimited = False
#num_epochs = 99999 if Unlimited is False else math.inf

def generateDataSet(data_url=DEFAULT_DATA_URL):
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

def generateResults(train=False, num_epochs=10):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ganModelNames = ['Vanila-Base-Gan', 'SN-Gan', 'W-Gan', 'SN-W-Gan']
    resultsDirPath = 'project/final-results/'
    if train:
        trainModels(ganModelNames, num_epochs)
        resultsDirPath = 'project/results/'
    print('Generating Results')
    for name in ganModelNames:
            modelPath = resultsDirPath + name
            generateResultsFromPath(name, modelPath)
    print('Inception Score Comparition:')
    plotScore(resultsDirPath, ganModelNames, num_epochs)


def plotScore(path, ganModelNames, num_epochs):

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

def trainModels(ganModelNames, num_epochs=10):

    ds_gwb = generateDataSet()
    x0, y0 = ds_gwb[0]
    x0 = x0.unsqueeze(0).to(device)

    batch_size = DLparams['batch_size']
    im_size = ds_gwb[0][0].shape

    dl_train = DataLoader(ds_gwb, batch_size, shuffle=True)

    # constract new models
    ganModels = dict()
    if 'Vanila-Base-Gan' in ganModelNames:
        ganModels['Vanila-Base-Gan'] = Gan(im_size, device=device)
    if 'SN-Gan' in ganModelNames:
        ganModels['SN-Gan'] = SnGan(im_size, device=device)
    if 'W-Gan' in ganModelNames:
        ganModels['W-Gan'] = WGan(im_size, device=device)
    if 'SN-W-Gan' in ganModelNames:
        ganModels['SN-W-Gan'] = SnWGan(im_size, device=device)
    # train
    try:
        model_file_path = {}
        for ganName, ganModule in ganModels.items():
            model_file_path[ganName] = pathlib.Path().parent.absolute().joinpath(
                f'project/results/{ganName}/')
            if not os.path.exists(model_file_path[ganName]):
                os.makedirs(model_file_path[ganName])

        scores, dsc_avg_losses, gen_avg_losses ={}, {}, {}
        for ganName, ganModule in ganModels.items():
            dsc_avg_losses[ganName] = []
            gen_avg_losses[ganName] = []
            scores[ganName] = []

        for epoch_idx in range(num_epochs):
            # We'll accumulate batch losses and show an average once per epoch.
            dsc_losses, gen_losses = {}, {}
            for ganName, ganModule in ganModels.items():
                dsc_losses[ganName] = []
                gen_losses[ganName] = []
            print(f'--- EPOCH {epoch_idx + 1}/{num_epochs} ---')

            with tqdm.tqdm(total=len(dl_train.batch_sampler), file=sys.stdout) as pbar:
                for batch_idx, (x_data, _) in enumerate(dl_train):
                    x_data = x_data.to(device)
                    for ganName, ganModule in ganModels.items():
                        dsc_loss, gen_loss = ganModule.trainBatch(x_data)
                        dsc_losses[ganName].append(dsc_loss)
                        gen_losses[ganName].append(gen_loss)
                    pbar.update()
            for ganName, ganModule in ganModels.items():
                gen = ganModule.generator
                dsc_avg_losses[ganName].append(np.mean(dsc_losses[ganName]))
                gen_avg_losses[ganName].append(np.mean(gen_losses[ganName]))
                if epoch_idx % step == 0:
                    print(f'{ganName}')
                    print(f'Discriminator loss: {dsc_avg_losses[ganName][-1]}')
                    print(f'Generator loss:     {gen_avg_losses[ganName][-1]}')

            for ganName, ganModule in ganModels.items():
                if epoch_idx % step == 0:
                    print(f'========{ganName}========')
                    gen = ganModule.generator
                    samples = gen.sample(5, with_grad=False)
                    fig, _ = plot.tensors_as_images(samples.cpu(), figsize=(6, 2))
                    # torch.save(fig, model_file_path[ganName].joinpath('fig.pt'))
                    IPython.display.display(fig)
                    plt.close(fig)
                    print(f'========================')
                    scores[ganName] += [inception_score(gen, cuda=True, batch_size=32, resize=True, splits=10, len=20000)]
                    torch.save(scores[ganName], model_file_path[ganName].joinpath('scores.pt'))
                    print(f'score for{ganName} is mean: {scores[ganName][-1][0]} std:{scores[ganName][-1][1]} ' )


        ganModels = {ganName: ganModule.generator for ganName, ganModule in ganModels.items()}
    except KeyboardInterrupt as e:
        # for ganName, ganModel in ganModels.items():
        #     gen = ganModel.generator
        #     samples = gen.sample(9, with_grad=False)
        #     # fig, _ = plot.tensors_as_images(samples.cpu(), figsize=(6, 2))
        #     torch.save(samples, model_file_path[ganName].joinpath('generated_samples.pt'))
        #     # IS = inception_score(gen, cuda=True, batch_size=32, resize=True, splits=10, len=50000)
        #     # with open(model_file_path[ganName].joinpath('inception_score.txt'), 'w') as f:
        #     #     f.write(str(IS))
        print('\n *** Training interrupted by user')

    for ganName, gen in ganModels.items():
        samples = gen.sample(50, with_grad=False)
        # fig, _ = plot.tensors_as_images(samples.cpu(), figsize=(10, 2))
        torch.save(samples, model_file_path[ganName].joinpath('generated_samples.pt'))
        # IS = inception_score(gen, cuda=True, batch_size=32, resize=True, splits=10, len=50000)
        # with open(model_file_path[ganName].joinpath('inception_score.txt'), 'w') as f:
        #     f.write(str(IS))
        # print(f"Inception score for model - is: {IS}")
    print('Training Complete')



def generateResultsFromPath(modelName, modelPath):
    print('=====================================')
    print(f'Model Type: {modelName}')
    # with open(f'{modelPath}/inception_score.txt', 'r') as f:
    #     print(f'Inception Score: {f.read()}')
    print(f'Generated Images: ')
    samples = torch.load(f'{modelPath}/generated_samples.pt')
    fig, _ = plot.tensors_as_images(samples.cpu(), figsize=(15,10), nrows=5)
    IPython.display.display(fig)
    plt.close(fig)
    print('=====================================')