#
# Copyright 2016 The BigDL Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#


import matplotlib
matplotlib.use("Agg")

from optparse import OptionParser
from bigdl.models.lenet.utils import *
from bigdl.dataset.transformer import *
from bigdl.nn.layer import *
from bigdl.nn.criterion import *
from bigdl.optim.optimizer import *
from bigdl.util.common import *
import datetime as dt
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow


def build_model(class_num, k1, k2):
    model = Sequential()
    model.add(Reshape([1, 28, 28]))
    model.add(SpatialConvolution(1, 6, k1, k1))
    model.add(Tanh())
    model.add(SpatialMaxPooling(2, 2, 2, 2))
    model.add(SpatialConvolution(6, 12, k2, k2))
    model.add(Tanh())
    model.add(SpatialMaxPooling(2, 2, 2, 2))
    model.add(Reshape([12 *((31-k1-2*k2)/4)* ((31-k1-2*k2)/4)]))
    model.add(Linear(12 * ((31-k1-2*k2)/4) * ((31-k1-2*k2)/4), 100))
    model.add(Tanh())
    model.add(Linear(100, class_num))
    model.add(LogSoftMax())
    return model


if __name__ == "__main__":
    parser = OptionParser()
    parser.add_option("-a", "--action", dest="action", default="train")
    parser.add_option("-b", "--batchSize", type=int, dest="batchSize", default="128")
    parser.add_option("-o", "--modelPath", dest="modelPath", default="/tmp/lenet5/model.470")
    parser.add_option("-c", "--checkpointPath", dest="checkpointPath", default="/tmp/lenet5")
    parser.add_option("-t", "--endTriggerType", dest="endTriggerType", default="epoch")
    parser.add_option("-n", "--endTriggerNum", type=int, dest="endTriggerNum", default="20")
    parser.add_option("-d", "--dataPath", dest="dataPath", default="/tmp/mnist")
    parser.add_option("-l", "--learningRate", dest="learningRate", default="0.01")
    parser.add_option("-k", "--learningrateDecay", dest="learningrateDecay", default="0.0002")
    parser.add_option("-s", "--kernelSize1", type=int, dest="kernelSize1", default="3")
    parser.add_option("-e", "--kernelSize2", type=int, dest="kernelSize2", default="3")
    (options, args) = parser.parse_args(sys.argv)

    sc = SparkContext(appName="lenet5", conf=create_spark_conf())
    redire_spark_logs()
    show_bigdl_info_logs()
    init_engine()
    learning_rate=float(options.learningRate)
    learning_rate_decay=float(options.learningrateDecay)
    print(learning_rate)
    if options.action == "train":
        (train_data, test_data) = preprocess_mnist(sc, options)

        optimizer = Optimizer(
            model=build_model(10,options.kernelSize1,options.kernelSize2),
            training_rdd=train_data,
            criterion=ClassNLLCriterion(),
            optim_method=SGD(learningrate=learning_rate, learningrate_decay=learning_rate_decay),
            end_trigger=get_end_trigger(options),
            batch_size=options.batchSize)

        validate_optimizer(optimizer, test_data, options)
        app_name = "lenet5" + dt.datetime.now().strftime("%Y%m%d-%H%M%S")
        train_summary_instance = TrainSummary(log_dir="/home/am72ghiassi/bd/result_log/", app_name=app_name)

        val_summary_instance = ValidationSummary(log_dir='/home/am72ghiassi/bd/result_log/',
                                                 app_name=app_name)
        optimizer.set_train_summary(train_summary_instance)
        optimizer.set_val_summary(val_summary_instance)
        print("saving logs to ", app_name)

        trained_model = optimizer.optimize()
        print("datapath:", options.dataPath, "\n", "kernelsize:", options.kernelSize1, "\n",
              "kernelsize2:", options.kernelSize2, "epoch:", options.endTriggerNum)
        
        # read the summary of train and test logs
        train_loss = np.array(train_summary_instance.read_scalar("Loss"))
        val_top1 = np.array(val_summary_instance.read_scalar("Top1Accuracy"))
        train_throughput = np.array(train_summary_instance.read_scalar("Throughput"))

        print("trainloss:", train_loss, "\n", "val_top1:", val_top1, "\n", "train_throughput:", train_throughput)


        # Plot figures of the train_loss-step, val_acc-step, train_throughput-step
        # plt.figure(figsize=(15, 15))
        # plt.subplot(2, 2, 1)
        # plt.plot(train_loss[:, 0], train_loss[:, 1], label='train_loss')
        # plt.xlim(0, train_loss.shape[0] + 10)
        # plt.grid(True)
        # plt.xlabel("step")
        # plt.ylabel("train_loss")
        # plt.title("train_loss of {datapath}, ks1{ks1}, ks2{ks2}, epoch{epoch}.png".format(datapath=options.dataPath, ks1=options.kernelSize1,
        #                                                                    ks2=options.kernelSize2, epoch=options.endTriggerNum))
        # plt.subplot(2, 2, 2)
        # plt.plot(val_top1[:, 0], val_top1[:, 1], label='val_top1')
        # plt.xlim(0, val_top1.shape[0] + 10)
        # plt.title("validation top1accuracy")
        # plt.grid(True)
        # plt.subplot(2, 2, 3)
        # plt.plot(train_throughput[:, 0], train_throughput[:, 1], label='train_throughput')
        # plt.xlim(0, train_throughput.shape[0] + 10)
        # plt.title("train_throughput")
        # plt.grid(True)
        #
        # plt.savefig("/home/am72ghiassi/bd/fig/" + options.kernelSize1 + options.kernelSize2 + options.endTriggerNum+".png")
        
    elif options.action == "test":
        # Load a pre-trained model and then validate it through top1 accuracy.
        test_data = get_mnist(sc, "test", options.dataPath) \
            .map(lambda rec_tuple: (normalizer(rec_tuple[0], mnist.TEST_MEAN, mnist.TEST_STD),
                                    rec_tuple[1])) \
            .map(lambda t: Sample.from_ndarray(t[0], t[1]))
        model = Model.load(options.modelPath)
        print("this is test result of ", options.modelPath)
        results = model.evaluate(test_data, options.batchSize, [Top1Accuracy()])
        for result in results:
            print(result)
    
    sc.stop()

