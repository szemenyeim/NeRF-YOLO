import torch
from data.nerfDataSet import NeRFDataSet
from torch.utils.data import DataLoader
from models.nerfAttention import NeRFAttention, SmoothNLLLoss
import tqdm

if __name__ == '__main__':

    lr = 1e-3
    wd = 1e-5
    numEpoch = 100

    device = "cuda"

    path = "./nerfData/"

    trSS = NeRFDataSet(path, "train", "640")
    trSM = NeRFDataSet(path, "train", "1280")
    trSL = NeRFDataSet(path, "train", "2560")
    valSS = NeRFDataSet(path, "val", "640")
    valSM = NeRFDataSet(path, "val", "1280")
    valSL = NeRFDataSet(path, "val", "2560")

    bSizes = [16,2,1]

    trLS = DataLoader(trSS, bSizes[0], True)
    trLM = DataLoader(trSM, bSizes[1], True)
    trLL = DataLoader(trSL, bSizes[2], True)
    valLS = DataLoader(valSS, bSizes[0], True)
    valLM = DataLoader(valSM, bSizes[1], True)
    valLL = DataLoader(valSL, bSizes[2], True)

    model = NeRFAttention(nPos=6, nEmbed=32).cuda()

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, numEpoch, lr)

    #criterion = torch.nn.SmoothL1Loss(beta=0.1).cuda()
    criterion = torch.nn.BCELoss()
    criterion = SmoothNLLLoss()


    bestLoss = 100000

    scale = torch.tensor([
        [1., 0., 0.],
        [0., 1., 0.],
        [0., 0., 1.]
    ]).cuda()

    features = None

    for epoch in range(numEpoch):
        rLossS = 0
        rLossM = 0
        rLossL = 0

        model.train()

        for inS, outS in tqdm.tqdm(trLS):
        #for (inS, outS), (inM, outM), (inL, outL) in tqdm.tqdm(zip(trLS, trLM, trLL)):

            inS, outS = inS.cuda(), outS.cuda()
            #inM, outM = inM.cuda(), outM.cuda()
            #inL, outL = inL.cuda(), outL.cuda()

            _, predS = model(features, inS, scale[0])
            #_, predM = model(features, inM, scale[1])
            #_, predL = model(features, inL, scale[2])

            lossS = criterion(predS, outS)
            #lossM = criterion(predM, outM)
            #lossL = criterion(predL, outL)

            lossT = lossS# + lossM + lossL

            lossT.backward()
            optimizer.step()

            rLossS += lossS.item()
            #rLossM += lossM.item()
            #rLossL += lossL.item()

        tLoss = (rLossS)/len(trLS)#+rLossM+rLossL
        print("Train epoch %d finished: Loss: %.4f, small: %.4f, medium: %.4f, large: %.4f" % (epoch+1, tLoss, rLossS, rLossM, rLossL))

        rLossS = 0
        rLossM = 0
        rLossL = 0

        model.eval()

        for inS, outS in tqdm.tqdm(valLS):
        #for (inS, outS), (inM, outM), (inL, outL) in tqdm.tqdm(zip(valLS, valLM, valLL)):
            inS, outS = inS.cuda(), outS.cuda()
            #inM, outM = inM.cuda(), outM.cuda()
            #inL, outL = inL.cuda(), outL.cuda()

            _, predS = model(features, inS, scale[0])
            #_, predM = model(features, inM, scale[1])
            #_, predL = model(features, inL, scale[2])

            lossS = criterion(predS, outS)
            #lossM = criterion(predM, outM)
            #lossL = criterion(predL, outL)

            lossT = lossS #+ lossM + lossL

            rLossS += lossS.item()
            #rLossM += lossM.item()
            #rLossL += lossL.item()

        tLoss = (rLossS)/len(valLS) #+ rLossM + rLossL
        print("Val epoch %d finished: Loss: %.4f, small: %.4f, medium: %.4f, large: %.4f" % (
        epoch + 1, tLoss, rLossS, rLossM, rLossL))

        scheduler.step()

        if bestLoss > tLoss:
            print("Best loss achieved")
            bestLoss = tLoss
            torch.save(model, "./models/bestAtt.pt")