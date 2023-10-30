#-------------------------------------#
#       对数据集进行训练
#-------------------------------------#
import os
from config.opts import opts
import numpy as np
import time
import torch
from torch.autograd import Variable
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from net.FBODInferenceNet import FBODInferenceBody_MultiScale
from utils.FBODLoss import LossFuncM
from FB_detector_multi_scale import FB_Postprocess
from tqdm import tqdm
import matplotlib.pyplot as plt
from utils.utils import FBObj
from dataloader.mydataset import CustomDataset_multi_head, dataset_collate_multi_head
from mAP import mean_average_precision
import copy

os.environ['KMP_DUPLICATE_LIB_OK']='True'

#---------------------------------------------------#
#   获得类
#---------------------------------------------------#
def get_classes(classes_path):
    '''loads the classes'''
    with open(classes_path) as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]
    return class_names

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

class LablesToResults(object):
    def __init__(self, batch_size):#h,w
        self.batch_size = batch_size

    def covert(self, labels_list, iteration): # TO Raw image size
        label_obj_list = []
        for batch_id in range(self.batch_size):
            labels = labels_list[batch_id]
            if labels.size==0:
                continue
            image_id = self.batch_size*iteration + batch_id
            for label in labels:
                # class_id = label[4] + 1 ###Include background in this project, the label didn't include background classes.
                box = [label[i] for i in range(4)]
                label_obj_list.append(FBObj(score=1., image_id=image_id, bbox=box))
        return label_obj_list

def fit_one_epoch(largest_AP_50,net,loss_func,epoch,epoch_size,epoch_size_val,gen,genval,Epoch,cuda,save_model_dir,labels_to_results,detect_post_process):
    total_loss = 0
    val_loss = 0
    start_time = time.time()
    with tqdm(total=epoch_size,desc=f'Epoch {epoch + 1}/{Epoch}',postfix=dict,mininterval=0.3) as pbar:
        for iteration, batch in enumerate(gen):
            if iteration >= epoch_size:
                break
            images, targets= batch[0], batch[1]
            ### targets [[large],[medium],[small]]
            # print(len(targets[0][0]))
            # print(targets[1][0][0].size())
            #print(images.shape) 1,7,384,672
            with torch.no_grad():
                # if cuda:
                #     images = Variable(images).to(torch.device('cuda:0'))
                #     targets = [Variable(fature_label.to(torch.device('cuda:0'))) for fature_label in targets] ## 
                # else:
                #     images = Variable(images)
                #     targets = [Variable(fature_label)for fature_label in targets] ##
                if cuda:
                    images = Variable(torch.from_numpy(images)).to(torch.device('cuda:0'))
                    targets = [[Variable(torch.from_numpy(fature_label)) for fature_label in ann] for ann in targets] ##
                else:
                    images = Variable(torch.from_numpy(images))
                    targets = [[Variable(torch.from_numpy(fature_label).type(torch.FloatTensor)) for fature_label in ann] for ann in targets] ##
            optimizer.zero_grad()
            outputs = net(images)
            losses = []
            for i in range(3):
                loss_item = loss_func[i](outputs[i], targets[i])
                losses.append(loss_item)
            loss = sum(losses)
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                total_loss += loss
            waste_time = time.time() - start_time
            
            pbar.set_postfix(**{'total_loss': total_loss.item() / (iteration + 1), 
                                'lr'        : get_lr(optimizer),
                                'step/s'    : waste_time})
            pbar.update(1)

            start_time = time.time()
    net.eval()
    print('Start Validation')
    with tqdm(total=epoch_size_val, desc=f'Epoch {epoch + 1}/{Epoch}',postfix=dict,mininterval=0.3) as pbar:
        all_label_obj_list = []
        all_obj_result_list = []
        for iteration, batch in enumerate(genval):
            if iteration >= epoch_size_val:
                break
            images_val, targets_val, labels_list = batch[0], batch[1], batch[2]
            with torch.no_grad():
                # if cuda:
                #     images_val = Variable(images_val).to(torch.device('cuda:0'))
                #     targets_val = [Variable(fature_label.to(torch.device('cuda:0'))) for fature_label in targets_val] ## 
                # else:
                #     images_val = Variable(images_val)
                #     targets_val = [Variable(fature_label)for fature_label in targets_val] ##
                if cuda:
                    images_val = Variable(torch.from_numpy(images_val)).to(torch.device('cuda:0'))
                    targets_val = [[Variable(torch.from_numpy(fature_label)) for fature_label in ann] for ann in targets_val] ##
                else:
                    images_val = Variable(torch.from_numpy(images_val))
                    targets_val = [[Variable(torch.from_numpy(fature_label).type(torch.FloatTensor)) for fature_label in ann] for ann in targets_val] ##
                optimizer.zero_grad()
                outputs = net(images_val)

                losses = []
                for i in range(3):
                    loss_item = loss_func[i](outputs[i], targets_val[i])
                    losses.append(loss_item)
                loss = sum(losses)
                val_loss += loss

                if (epoch+1) >= 30:
                    label_obj_list = labels_to_results.covert(labels_list, iteration)
                    all_label_obj_list += label_obj_list

                    obj_result_list = detect_post_process.Process(outputs, iteration)
                    all_obj_result_list += obj_result_list

            pbar.set_postfix(**{'total_loss': val_loss.item() / (iteration + 1)})
            pbar.update(1)
    net.train()
    if (epoch+1) >= 30:
        print("here")
        AP_50,REC_50,PRE_50=mean_average_precision(all_obj_result_list,all_label_obj_list,iou_threshold=0.5)
    else:
        AP_50,REC_50,PRE_50 = 0,0,0
    
    print('Finish Validation')
    print('Epoch:'+ str(epoch+1) + '/' + str(Epoch))
    print('Total Loss: %.4f || Val Loss: %.4f  || AP_50: %.4f  || REC_50: %.4f  || PRE_50: %.4f' % (total_loss/(epoch_size+1), val_loss/(epoch_size_val+1),  AP_50, REC_50, PRE_50))
    
    if (epoch+1)%10 == 0 or epoch == 0:
        if largest_AP_50 < AP_50:
            largest_AP_50 = AP_50
        print('Saving state, iter:', str(epoch+1))
        torch.save(model.state_dict(), save_model_dir + 'Epoch%d-Total_Loss%.4f-Val_Loss%.4f-AP_50_%.4f.pth'%((epoch+1),total_loss/(epoch_size+1),val_loss/(epoch_size_val+1),AP_50))
        torch.save(model.state_dict(), save_model_dir + 'FB_object_detect_model.pth')
    else:
        if largest_AP_50 < AP_50:
            largest_AP_50 = AP_50
            print('Saving state, iter:', str(epoch+1))
            torch.save(model.state_dict(), save_model_dir + 'Epoch%d-Total_Loss%.4f-Val_Loss%.4f-AP_50_%.4f.pth'%((epoch+1),total_loss/(epoch_size+1),val_loss/(epoch_size_val+1),AP_50))
            torch.save(model.state_dict(), save_model_dir + 'FB_object_detect_model.pth')
    if (epoch+1) >= 30:
        return total_loss/(epoch_size+1), val_loss/(epoch_size_val+1), largest_AP_50, AP_50
    else:
        return total_loss/(epoch_size+1), val_loss/(epoch_size_val+1), largest_AP_50, 0.80

num_to_english_c_dic = {1:"one", 3:"three", 5:"five", 7:"seven", 9:"nine", 11:"eleven"}

####################### Plot figure #######################################
x_epoch = []
record_loss = {'train_loss':[], 'test_loss':[]}
fig = plt.figure()

ax0 = fig.add_subplot(111, title="Train the FB_object_detect model")
ax0.set_ylabel('loss')
ax0.set_xlabel('Epochs')

def draw_curve_loss(epoch, train_loss, test_loss, pic_name):
    global record_loss
    record_loss['train_loss'].append(train_loss)
    record_loss['test_loss'].append(test_loss)

    x_epoch.append(int(epoch))
    ax0.plot(x_epoch, record_loss['train_loss'], 'b', label='train')
    ax0.plot(x_epoch, record_loss['test_loss'], 'r', label='val')
    if epoch == 1:
        ax0.legend()
    fig.savefig(pic_name)
########============================================================########
x_ap50_epoch = []
record_ap50 = {'AP_50':[]}
fig_ap50 = plt.figure()

ax1 = fig_ap50.add_subplot(111, title="Train the FB_object_detect model")
ax1.set_ylabel('ap_50')
ax1.set_xlabel('Epochs')

def draw_curve_ap50(epoch, ap_50, pic_name):
    global record_ap50
    record_ap50['AP_50'].append(ap_50)

    x_ap50_epoch.append(int(epoch))
    ax1.plot(x_ap50_epoch, record_ap50['AP_50'], 'g', label='AP_50')
    if epoch == 30:
        ax1.legend()
    fig_ap50.savefig(pic_name)
#############################################################################

if __name__ == "__main__":

    opt = opts().parse()
    # assign_method: The label assign method. binary_assign, guassian_assign or auto_assign
    if opt.assign_method == "binary_assign":
        abbr_assign_method = "ba"
    elif opt.assign_method == "guassian_assign":
        abbr_assign_method = "ga"
    else:
        raise("Error! assign_method error.")
    
    save_model_dir = "logs/" + num_to_english_c_dic[opt.input_img_num] + "/" + opt.model_input_size + "/" + opt.input_mode + "_" + opt.aggregation_method \
                             + "_" + opt.backbone_name + "_" + opt.fusion_method + "_" + abbr_assign_method + "_"  + opt.Add_name + "/"
    os.makedirs(save_model_dir, exist_ok=True)

    ############### For log figure ################
    log_pic_name_loss = "train_output_img/" + num_to_english_c_dic[opt.input_img_num] + "/" +opt.model_input_size + "/" + opt.input_mode + "_" + opt.aggregation_method \
                                            + "_" + opt.backbone_name + "_" + opt.fusion_method + "_" + "loss_" + abbr_assign_method + "_" + opt.Add_name + ".jpg"
    log_pic_name_ap50 = "train_output_img/" + num_to_english_c_dic[opt.input_img_num] + "/" +opt.model_input_size + "/" + opt.input_mode + "_" + opt.aggregation_method \
                                            + "_" + opt.backbone_name + "_" + opt.fusion_method + "_" + "ap50_" + abbr_assign_method + "_" + opt.Add_name + ".jpg"
    os.makedirs("train_output_img/" + num_to_english_c_dic[opt.input_img_num] + "/" + opt.model_input_size + "/", exist_ok=True)
    ################################################

    #-------------------------------#
    #-------------------------------#
    model_input_size = (int(opt.model_input_size.split("_")[0]), int(opt.model_input_size.split("_")[1])) # H,W
    
    Cuda = True

    train_annotation_path = "./dataloader/" + "img_label_" + num_to_english_c_dic[opt.input_img_num] + "_continuous_difficulty_train.txt"
    train_dataset_image_path = opt.data_path + "train/images/"
    
    val_annotation_path = "./dataloader/" + "img_label_" + num_to_english_c_dic[opt.input_img_num] + "_continuous_difficulty_val.txt"
    val_dataset_image_path = opt.data_path + "val/images/"
    #-------------------------------#
    # 
    #-------------------------------#
    classes_path = 'model_data/classes.txt'   
    class_names = get_classes(classes_path)
    num_classes = len(class_names) + 1 #### Include background
    
    # create model
    ### FBODInferenceBody_MultiScale parameters:
    ### input_img_num=5, aggregation_output_channels=16, aggregation_method="multiinput", input_mode="GRG", ### Aggreagation parameters.
    ### backbone_name="cspdarknet53": ### Extract parameters. input_channels equal to aggregation_output_channels.
    model = FBODInferenceBody_MultiScale(input_img_num=opt.input_img_num, aggregation_output_channels=opt.aggregation_output_channels,
                                         aggregation_method=opt.aggregation_method, input_mode=opt.input_mode, backbone_name=opt.backbone_name, fusion_method=opt.fusion_method)

    #-------------------------------------------#
    #   load model
    #-------------------------------------------#
    if os.path.exists(opt.pretrain_model_path):
        print('Loading weights into state dict...')
        if Cuda:
            device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        else:
            device = torch.device('cpu')
        model_dict = model.state_dict()
        pretrained_dict = torch.load(opt.pretrain_model_path, map_location=device)
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if np.shape(model_dict[k]) ==  np.shape(v)}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
        print('Finished loading pretrained model!')
    else:
        print('Train the model from scratch!')

    net = model.train()

    if Cuda:
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = True
        net = net.cuda()

    # 建立loss函数
    # dynamic label assign, so the gettargets is ture.
    stride_list = [32, 8, 2]
    ### large, medium, small
    scale_min_max_list = [[80,256],[48,80],[13,48]] ### Need to count the object scale, and divide to 3 sets, log the min max of each set.
    loss_funcM = []
    for i in range(3):
        loss_funcM.append(LossFuncM(num_classes=num_classes, cuda=Cuda, scale_list=scale_min_max_list[i], stride=stride_list[i], gettargets=False))

    # For calculating the AP50
    scale_max_list = [scale_min_max_list[0][1], scale_min_max_list[1][1], scale_min_max_list[2][1]]
    detect_post_process = FB_Postprocess(batch_size=opt.Batch_size, model_input_size=model_input_size, scale_max_list=scale_max_list)
    labels_to_results = LablesToResults(batch_size=opt.Batch_size)

    # # 0.2用于验证，0.8用于训练
    # val_split = 0.1
    # with open(train_annotation_path) as f:
    #     lines = f.readlines()
    # np.random.seed(10101)
    # np.random.shuffle(lines)
    # np.random.seed(None)
    # num_val = int(len(lines)*val_split)
    # num_train = len(lines) - num_val

    with open(train_annotation_path) as f:
        train_lines = f.readlines()
        num_train = len(train_lines)
    with open(val_annotation_path) as f:
        val_lines = f.readlines()
        num_val = len(val_lines)

    
    #------------------------------------------------------#
    #------------------------------------------------------#
    lr = 1e-3
    Batch_size = opt.Batch_size
    Freeze_Epoch = 0
    # Freeze_Epoch = 85
    Unfreeze_Epoch = 100
    # Unfreeze_Epoch = 200

    optimizer = optim.Adam(net.parameters(),lr,weight_decay=5e-4)
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer,step_size=1,gamma=0.95)
    
    train_data = CustomDataset_multi_head(train_lines, (model_input_size[1], model_input_size[0]), image_path=train_dataset_image_path, input_mode=opt.input_mode, continues_num=opt.input_img_num)
    train_dataloader = DataLoader(train_data, batch_size=Batch_size, shuffle=True, num_workers=4, pin_memory=True, collate_fn=dataset_collate_multi_head)
    # train_dataloader = DataLoader(train_data, batch_size=Batch_size, shuffle=True, num_workers=4, pin_memory=True)
    
    val_data = CustomDataset_multi_head(val_lines, (model_input_size[1], model_input_size[0]), image_path=val_dataset_image_path, input_mode=opt.input_mode, continues_num=opt.input_img_num)
    val_dataloader = DataLoader(val_data, batch_size=Batch_size, shuffle=True, num_workers=4, pin_memory=True, collate_fn=dataset_collate_multi_head)
    # val_dataloader = DataLoader(val_data, batch_size=Batch_size, shuffle=True, num_workers=4, pin_memory=True)


    epoch_size = max(1, num_train//Batch_size)
    epoch_size_val = num_val//Batch_size
    #------------------------------------#
    #   解冻后训练
    #------------------------------------#
    for param in model.extract_features.backbone.parameters():
        param.requires_grad = True

    largest_AP_50=0
    for epoch in range(Freeze_Epoch,Unfreeze_Epoch):
        train_loss, val_loss,largest_AP_50_record, AP_50 = fit_one_epoch(largest_AP_50,net,loss_funcM,epoch,epoch_size,epoch_size_val,train_dataloader,val_dataloader,Unfreeze_Epoch,Cuda,save_model_dir, labels_to_results=labels_to_results, detect_post_process=detect_post_process)
        largest_AP_50 = largest_AP_50_record
        if (epoch+1)>=2:
            draw_curve_loss(epoch+1, train_loss.item(), val_loss.item(), log_pic_name_loss)
        if (epoch+1)>=30:
            draw_curve_ap50(epoch+1, AP_50, log_pic_name_ap50)
        lr_scheduler.step()