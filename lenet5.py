#!/usr/bin/env python
# coding: utf-8

# In[ ]:


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

from optparse import OptionParser
from bigdl.models.lenet.utils import *
from bigdl.dataset.transformer import *
from bigdl.nn.layer import *
from bigdl.nn.criterion import *
from bigdl.optim.optimizer import *
from bigdl.util.common import *
import datetime as dt


# k1 is the kernel size of convolution1, k2 is the kernel size of convolution2
def build_model(class_num, k1, k2): 
    model = Sequential()
    model.add(Reshape([1, 28, 28]))
    model.add(SpatialConvolution(1, 6, k1, k1))
    model.add(Tanh())
    model.add(SpatialMaxPooling(2, 2, 2, 2))
    model.add(SpatialConvolution(6, 12, k2, k2))
    model.add(Tanh())
    model.add(SpatialMaxPooling(2, 2, 2, 2))
    model.add(Reshape([12 * 4 * 4]))
    model.add(Linear(12 * 4 * 4, 100))
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
            model=build_model(10, options.kernelSize1, options.kernelSize2), #add the options to tune kernelSize
            training_rdd=train_data,
            criterion=ClassNLLCriterion(),
            optim_method=SGD(learningrate=learning_rate, learningrate_decay=learning_rate_decay),
            end_trigger=get_end_trigger(options),
            batch_size=options.batchSize)
        validate_optimizer(optimizer, test_data, options)
        trained_model = optimizer.optimize()
        parameters = trained_model.parameters()
        
        #read the summary of train and test logs
        train_summary_instance = TrainSummary(log_dir="/home/am72ghiassi/bd/result_log/", app_name="lenet5"+dt.datetime.now().strftime("%Y%m%d-%H%M%S"))
        train_summary_instance.set_summary_trigger("Parameters", SeveralIteration(50))
    
        val_summary_instance = ValidationSummary(log_dir='/home/am72ghiassi/bd/result_log/',
                                        app_name="lenet5"+dt.datetime.now().strftime("%Y%m%d-%H%M%S"))
        optimizer.set_train_summary(train_summary_instance)
        optimizer.set_val_summary(val_summary_instance)
        print ("saving logs to ","lenet5"+dt.datetime.now().strftime("%Y%m%d-%H%M%S"))
       
        #convert the useful results to array format
        train_loss = np.array(train_summary_instance.read_scalar("Loss"))
        val_top1 = np.array(val_summary_instance.read_scalar("Top1Accuracy"))
        train_throughput = np.array(train_summary_instance.read_scalar("Throughput"))
        val_throughput = np.array(val_summary_instance.read_scalar("Throughput"))
        
        print("trainloss:",train_loss,"\n", "val_top1:", val_top1, "\n", "train_throughput:", train_throughput,
             "\n", "val_throughput:", val_throughput)
        
        
        
    elif options.action == "test":
        # Load a pre-trained model and then validate it through top1 accuracy.
        test_data = get_mnist(sc, "test", options.dataPath)             .map(lambda rec_tuple: (normalizer(rec_tuple[0], mnist.TEST_MEAN, mnist.TEST_STD),
                                    rec_tuple[1])) \
            .map(lambda t: Sample.from_ndarray(t[0], t[1]))
        model = Model.load(options.modelPath)
        results = model.evaluate(test_data, options.batchSize, [Top1Accuracy()])
        for result in results:
            print(result)
    sc.stop()

