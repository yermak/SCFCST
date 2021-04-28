import torch
import json

from torchvision import datasets, transforms
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, IterableDataset
import glob
from torch import optim

cache = {}
combinations = (
    ('Harrier80', 'storm80', 'DEFT80', 'MAxWellMax', 'shap', 'Krichman'),
    ('Harrier80', 'storm80', 'MAxWellMax', 'DEFT80', 'shap', 'Krichman'),
    ('Harrier80', 'storm80', 'shap', 'DEFT80', 'MAxWellMax', 'Krichman'),
    ('Harrier80', 'storm80', 'Krichman', 'DEFT80', 'MAxWellMax', 'shap'),
    ('Harrier80', 'DEFT80', 'MAxWellMax', 'storm80', 'shap', 'Krichman'),
    ('Harrier80', 'DEFT80', 'shap', 'storm80', 'MAxWellMax', 'Krichman'),
    ('Harrier80', 'DEFT80', 'Krichman', 'storm80', 'MAxWellMax', 'shap'),
    ('Harrier80', 'MAxWellMax', 'shap', 'storm80', 'DEFT80', 'Krichman'),
    ('Harrier80', 'MAxWellMax', 'Krichman', 'storm80', 'DEFT80', 'shap'),
    ('Harrier80', 'shap', 'Krichman', 'storm80', 'DEFT80', 'MAxWellMax'),
)
apms = {'Harrier80': 42, 'storm80': 40, 'DEFT80': 90, 'MAxWellMax': 85, 'shap': 50, 'Krichman': 69}
races = {'Harrier80': 2, 'storm80': 2, 'DEFT80': 2, 'MAxWellMax': 1, 'shap': 0, 'Krichman': 1}


class FakeDataset(IterableDataset):
    # Zerg=0
    # Terran=1
    # Protoss = 2
    def __init__(self):
        self.order = combinations[0]
        self.options = []

        for row in combinations:
            sample = {}
            for p in self.order:
                if row.index(p) < 3:
                    team = 1
                else:
                    team = 2
                sample[p] = {'Team': team, 'Race': races[p], 'APM': apms[p] / 60}
            self.options.append(sample)

    def __iter__(self):
        for p in self.options:
            apm_ = (p['storm80']['Team'], p['storm80']['Race'], p['storm80']['APM'],
                    p['Harrier80']['Team'], p['Harrier80']['Race'], p['Harrier80']['APM'],
                    p['DEFT80']['Team'], p['DEFT80']['Race'], p['DEFT80']['APM'],
                    p['MAxWellMax']['Team'], p['MAxWellMax']['Race'], p['MAxWellMax']['APM'],
                    p['shap']['Team'], p['shap']['Race'], p['shap']['APM'],
                    p['Krichman']['Team'], p['Krichman']['Race'], p['Krichman']['APM']
                    )
            # print(torch.FloatTensor(apm_))
            yield torch.FloatTensor(apm_), 0


class JsonDataset(IterableDataset):

    def __init__(self, files):
        self.files = files

    def __iter__(self):
        for json_file in self.files:
            if json_file in cache:
                sample = cache[json_file]
            else:
                with open(json_file) as f:
                    sample = json.load(f)
                    cache[json_file] = sample

            players = sample['Header']['Players']
            if len(players) == 6:
                players_stats = {}

                for apm in sample['Computed']['PlayerDescs']:
                    players_stats[apm['PlayerID']] = apm['APM']

                # print(players_stats)

                p = {}
                for player in players:
                    p[player['Name']] = {'ID': player['ID'], 'Team': player['Team'], 'Race': player['Race']['ID'],
                                         'APM': players_stats[player['ID']] / 60}
                # print(p)

                if 'storm80' in p and 'Harrier80' in p and 'DEFT80' in p and 'MAxWellMax' in p and 'shap' in p and 'Krichman' in p:
                    apm_ = (
                        p['Harrier80']['Team'], p['Harrier80']['Race'], p['Harrier80']['APM'],
                        p['storm80']['Team'], p['storm80']['Race'], p['storm80']['APM'],
                        p['DEFT80']['Team'], p['DEFT80']['Race'], p['DEFT80']['APM'],
                        p['MAxWellMax']['Team'], p['MAxWellMax']['Race'], p['MAxWellMax']['APM'],
                        p['shap']['Team'], p['shap']['Race'], p['shap']['APM'],
                        p['Krichman']['Team'], p['Krichman']['Race'], p['Krichman']['APM']
                    )
                    # print(torch.FloatTensor(apm_))
                    yield torch.FloatTensor(apm_), sample['Computed']['WinnerTeam']


class SCFCST(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(18, 12)
        self.fc2 = nn.Linear(12, 6)
        self.fc3 = nn.Linear(6, 3)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        x = F.log_softmax(x, dim=1)
        return x


if __name__ == '__main__':
    print('CUDA is enabled: ', torch.cuda.is_available())

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    mean, std = (0.5,), (0.5,)

    # Create a transform and normalise data
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize(mean, std)
                                    ])

    dataset = JsonDataset(glob.glob("data/*.json"))
    dataloader = DataLoader(dataset, batch_size=10)

    testset = JsonDataset(glob.glob("test/*.json"))
    testloader = DataLoader(testset, batch_size=1)

    fakeset = FakeDataset()
    fakeloader = DataLoader(fakeset, batch_size=1)

    model = SCFCST()

    model.to(device)

    criterion = nn.NLLLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01)

    num_epochs = 1000

    for i in range(num_epochs):
        cum_loss = 0

        for games, winners in dataloader:
            games = games.to(device)
            winners = winners.to(device)
            optimizer.zero_grad()
            output = model(games)
            loss = criterion(output, winners)
            loss.backward()
            optimizer.step()

            cum_loss += loss.item()

    model.to('cpu')

    with torch.no_grad():
        num_correct = 0
        total = 0

        # set_trace()
        for games, winners in testloader:
            logps = model(games)
            output = torch.exp(logps)

            pred = torch.argmax(output, 1)
            total += winners.size(0)
            num_correct += (pred == winners).sum().item()

        print(f'Accuracy of the model on the test games: {num_correct * 100 / total}% ')

    with torch.no_grad():
        num_correct = 0
        total = 0

        # set_trace()
        i = 0
        for games, winners in fakeloader:
            logps = model(games)
            output = torch.exp(logps)

            pred = torch.argmax(output, 1)
            total += winners.size(0)
            print('Option :', combinations[i], 'prediction:', output, ', most likely team:', pred)
            i+=1
