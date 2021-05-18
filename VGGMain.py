import VGGData 
import VGGModel
import VGGTrain

from glob import glob

class VGG_Main():
    def __init__(self):
        self.path = './drive/Shareddrives/zeogi_gogi/dataset/'
        self.DATA_PATH_TRAINING_LIST = glob(self.path+'train/*/*.jpg')
        self.DATA_PATH_TESTING_LIST = glob(self.path+'test/*/*.jpg')

    def main(self):
        trainloader, testloader = VGGData.dataloader(self.DATA_PATH_TRAINING_LIST, self.DATA_PATH_TESTING_LIST)
        cfg = { #8 + 3 =11 == vgg11
                'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
                # 10 + 3 = vgg 13
                'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
                #13 + 3 = vgg 16
                'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
                # 16 +3 =vgg 19
                'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'] }

        feature = VGGModel.make_layers(cfg['A'], batch_norm=True)
        model = VGGModel.Vgg(feature, num_classes=7, init_weights=True)
        print('-------------- MODEL --------------')
        print(model, '\n')

        print('-------------- TRAINING --------------')
        VGGTrain.VGGTrain(model, trainloader, testloader)


if __name__ == '__main__':
    main_class = VGG_Main()
    main_class.main()