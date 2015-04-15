
require 'dp'
require 'save_report'
require 'insects'

require 'custom_datasource_factory'
require 'custom_experiment_factory'


local UniformChoose = torch.class('dp.UniformChoose', 'dp.Choose')
UniformChoose.isUniformChoose = true

function UniformChoose:__init(config)
    self.low = config.low
    self.high = config.high
    self._options = config
end

function UniformChoose:sample()
    return (math.random() * (self.high - self.low) + self.low)
end

function UniformChoose:report()
    return {typename = torch.typename(self), options = self._options}
end


local UniformIntegerChoose = torch.class('dp.UniformIntegerChoose', 'dp.Choose')
UniformIntegerChoose.isUniformIntegerChoose = true

function UniformIntegerChoose:__init(config)
    self.low = config.low
    self.high = config.high
    self._options = config
end

function UniformIntegerChoose:sample()
    return(math.random(self.low, self.high))
end

function UniformIntegerChoose:report()
    return {typename = torch.typename(self), options = self._options}
end




--[[hyperparameters]]--
opt = {
   collection = 'insects',
   learningRate = dp.UniformChoose{low=0.1, high=0.8}, --training learning rate
   momentum = dp.UniformChoose{low=0.5, high=0.9}, --momentum factor to use for training
   maxOutNorm = dp.UniformChoose{low=0.5, high=3}, --maximum norm allowed for output neuron weights
   batchSize = dp.WeightedChoose{
         [32]=10, [64]=7, [128]=5, [256]=4, [16]=3, [8]=2, [512]=1 
        }, --number of examples per mini-batch
   maxTries = dp.UniformIntegerChoose{low=20, high=70}, --maximum number of epochs without reduction in validation error.
   maxEpoch = dp.UniformIntegerChoose{low=20, high=300}, --maximum number of epochs of training
   activation = 'ReLU', -- activation function
   kernelStride = {1, 1, 1},

   channelSize_1 = dp.WeightedChoose{[32]=10,[64]=5}, 
   channelSize_2 = dp.WeightedChoose{[64]=10,[128]=5},
   channelSize_3 = dp.WeightedChoose{[64]=10,[128]=5},

   padding = {0, 0, 0}, -- order padding to add to the input before performing the convolution
   kernelSize = {5, 5, 5}, -- size of the filters (filters are squares) for each conv layer
   kernelStride = {1, 1, 1}, -- stride of the filters in each conv layer 
   poolSize = {2, 2, 2}, 
   poolStride= {2, 2, 2},

   hiddenSize_1 = dp.UniformIntegerChoose({low=100, high=1000}),
   hiddenSize_2 = dp.UniformIntegerChoose({low=100, high=1000}),
   cuda = true,
   useDevice = 2,
   dropoutProb  = {0, 0, 0, 0.5, 0.5}, -- dropout probability of each layer
   dropout = true,
   visualize_dataset = true,
   standardize = true,
   zca = false, 
   lecunlcn = false,
   sparse_init = false,
   weight_decay_factor = 0.
}


    
input_preprocess = {}
if opt.standardize then
   table.insert(input_preprocess, dp.Standardize())
end
if opt.zca then
   table.insert(input_preprocess, dp.ZCA())
end
if opt.lecunlcn then
   table.insert(input_preprocess, dp.GCN())
   table.insert(input_preprocess, dp.LeCunLCN{progress=false})
end




print("loading dataset")
datasource = dp.Insects{input_preprocess = input_preprocess, labels_kind="order"}
print("dataset loaded")

--[[Model]]--



function build_model(opt)
    
    print("trying : ")
    print("==================================================")
    for i,v in pairs(opt) do 

        if type(v) == 'table' then
            content = ""
            for i, v in ipairs(v) do
                content = content..tostring(v)..","
            end
        else
            content = tostring(v)
        end
        print(i..":"..content)
    end


    opt.channelSize = {opt.channelSize_1, opt.channelSize_2, opt.channelSize_3}
    opt.hiddenSize = {opt.hiddenSize_1, opt.hiddenSize_2}

    function dropout(depth)
       return opt.dropout and (opt.dropoutProb[depth] or 0) > 0 and nn.Dropout(opt.dropoutProb[depth])
    end
    datasource = opt.datasource

    cnn = dp.Sequential()
    inputSize = datasource:imageSize('c')
    height, width = datasource:imageSize('h'), datasource:imageSize('w')
    depth = 1
    for i=1,#opt.channelSize do
        local conv = dp.Convolution2D{
            input_size = inputSize, 
            padding = opt.padding[i],
            kernel_size = {opt.kernelSize[i], opt.kernelSize[i]},
            kernel_stride = {opt.kernelStride[i], opt.kernelStride[i]},
            pool_size = {opt.poolSize[i], opt.poolSize[i]},
            pool_stride = {opt.poolStride[i], opt.poolStride[i]},
            output_size = opt.channelSize[i], 
            transfer = nn[opt.activation](),
            dropout = dropout(depth),
            sparse_input_preprocessinit = opt.sparse_init
       }
       cnn:add(conv)

       inputSize, height, width = conv:outputSize(height, width, 'bchw')
       depth = depth + 1
    end

    inputSize = inputSize*height*width
    print("input to first Neural layer has: "..inputSize.." neurons")

    for i,hiddenSize in ipairs(opt.hiddenSize) do
       local dense = dp.Neural{
          input_size = inputSize, 
          output_size = hiddenSize,
          transfer = nn[opt.activation](),
          dropout = dropout(depth),
          sparse_init = opt.sparse_init
       }
       cnn:add(dense)
       inputSize = hiddenSize
       depth = depth + 1
    end

    cnn:add(
       dp.Neural{
          input_size = inputSize, 
          output_size = #(datasource:classes()),
          transfer = nn.LogSoftMax(),
          dropout = dropout(depth),
          sparse_init = opt.sparse_init
       }
    )



    --[[Propagators]]--
    train = dp.Optimizer{
       loss = dp.NLL(),
       visitor = { -- the ordering here is important:
          dp.Momentum{momentum_factor = opt.momentum},
          dp.Learn{
                learning_rate = opt.learningRate,
                observer = {dp.AdaptiveLearningRate{}}
          },
          dp.MaxNorm{max_out_norm = opt.maxOutNorm},
          dp.WeightDecay{wd_factor = opt.weight_decay_factor}
       },
       feedback = dp.Confusion(),
       sampler = dp.ShuffleSampler{batch_size = opt.batchSize},
       progress = true
    }
    valid = dp.Evaluator{
       loss = dp.NLL(),
       feedback = dp.Confusion(),  
       sampler = dp.Sampler{}
    }
    test = dp.Evaluator{
       loss = dp.NLL(),
       feedback = dp.Confusion(),
       sampler = dp.Sampler{}
    }



    --[[Experiment]]--
    xp = dp.Experiment{
       model = cnn,
       optimizer = train,
       validator = valid,
       tester = test,
       observer = {
          dp.FileLogger(),
          dp.SaveReport(),
          dp.EarlyStopper{
             error_report = {'validator','feedback','confusion','accuracy'},
             maximize = true,
             max_epochs = opt.maxTries,
             save_strategy = dp.SaveToFile{},
          },
       },
       random_seed = os.time(),
       max_epoch = opt.maxEpoch
    }

    --[[GPU or CPU]]--
    if opt.cuda then
       require 'cutorch'
       require 'cunn'
       cutorch.setDevice(opt.useDevice)
       xp:cuda()
    end
    return(xp)
end

datasource_factory=dp.CustomDatasourceFactory{name="insects"}
experiment_factory=dp.CustomExperimentFactory{name="insects"}
opt.builder = build_model
opt.datasource = datasource

hyperopt = dp.HyperOptimizer{
      collection_name=opt.collection,
      hyperparam_sampler = dp.PriorSampler{--only samples random_seed
         name=opt.collection, dist=opt
      },
      experiment_factory = experiment_factory,
      datasource_factory=datasource_factory,
}

hyperopt:run()
--xp:run(datasource)
