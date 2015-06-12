-------------------------------------------------
--[[ CIFAR10 ]]--
-- http://www.cs.toronto.edu/~kriz/cifar.html
-- A color-image set of 10 different objects.
-- Small size makes it hard to generalize from train to test set.
-- Regime : overfitting.
-------------------------------------------------
--

local Insects, parent = torch.class("dp.Insects", "dp.DataSource")

function labels_to_indexes(filename)
    fd =  io.open(filename)
    mapping = {}
    labels = {}
    for line in fd:lines() do
        table.insert(labels, line)
        mapping[line] = true
    end
    i = 1
    classes = {}
    str_classes = {}
    for k, v in pairs(mapping) do
        mapping[k] = i
        classes[i] = i
        str_classes[i] = k
        i = i + 1
    end
    for i = 1, #labels do
        labels[i] = (mapping[labels[i]])
    end
    return labels, classes, str_classes
end


Insects.isInsects = true

Insects._name = 'insects'
--Insects._image_size = {3, 64, 64}
--Insects._feature_size = 3*64*64
Insects._image_axes = 'bchw'
--Insects._classes = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18}

function Insects:__init(config)
   config = config or {}
   assert(torch.type(config) == 'table' and not config[1], 
      "Constructor requires key-value arguments")
   self.args, 
   self._size,
   self._labels_kind,
   self._train_ratio, self._valid_ratio,
   self._scale,
   input_preprocess, target_preprocess
      = xlua.unpack( 
      {config},
      'Insects', nil,
      {arg='size', type='number', default=64},
      {arg='labels_kind', type='str', default='labels'},
      {arg='train_ratio', type='number', default=0.5,
        help='proportion of training set to use for cross-validation.'},
      {arg='valid_ratio', type='number', default=0.25,
        help='proportion of training set to use for cross-validation.'},
      {arg='scale', type='table', 
        help='bounds to scale the values between'},
      {arg='input_preprocess', type='table | dp.Preprocess',
        help='to be performed on set inputs, measuring statistics ' ..
        '(fitting) on the train_set only, and reusing these to ' ..
        'preprocess the valid_set and test_set.'},
      {arg='target_preprocess', type='table | dp.Preprocess',
        help='to be performed on set targets, measuring statistics ' ..
        '(fitting) on the train_set only, and reusing these to ' ..
        'preprocess the valid_set and test_set.'}  
   )
   if (self._scale == nil) then
      self._scale = {0,1}
   end
   self._targets, self._classes, self._str_classes = self:loadTargets()
   self._data = self:loadData()
   shuffle = torch.randperm(self._data:size(1)):long()

   self._data = self._data:index(1, shuffle)
   self._targets = self._targets:index(1, shuffle)
    
   self._image_size = {self._data:size(2), self._data:size(3), self._data:size(4)}
   self._feature_size = self._image_size[1]*self._image_size[2]*self._image_size[3]

   self._data = torch.reshape(self._data, self._data:size(1), self._data:size(2)*self._data:size(3)*self._data:size(4)  )
 

  self:loadTrain()
  self:loadValid()
  self:loadTest()
parent.__init(self, {train_set=self:trainSet(), 
                     valid_set=self:validSet(),
                     test_set=self:testSet(),
                     input_preprocess=input_preprocess,
                     target_preprocess=target_preprocess})
end

function Insects:loadTrain()
   --Data will contain a tensor where each row is an example, and where
   --the last column contains the target class.
   --
   local data = self._data
   local targets = self._targets

   local size = math.floor(data:size(1)*(self._train_ratio))
   local train_data = data:narrow(1, 1, size)
   local train_targets = targets:narrow(1, 1, size)
   self:setTrainSet(self:createDataSet(train_data, train_targets, 'train'))
   return self:trainSet()
end

function Insects:loadValid()
   local data = self._data
   local targets = self._targets

   local start = math.ceil(data:size(1)*(self._train_ratio))
   local size = math.ceil(data:size(1)*(self._valid_ratio))
   local valid_data = data:narrow(1, start, size)
   local valid_targets = targets:narrow(1, start, size)
   self:setValidSet(self:createDataSet(valid_data, valid_targets, 'valid'))
   return self:validSet()
end

function Insects:loadTest()

   local data = self._data
   local targets = self._targets

   local start = math.ceil(data:size(1)*(self._train_ratio + self._valid_ratio))
   local size = data:size(1) - start
   local test_data = data:narrow(1, start, size)
   local test_targets = targets:narrow(1, start, size)
   self:setTestSet(self:createDataSet(test_data, test_targets, 'test'))
   return self:testSet()
end

function Insects:createDataSet(data, targets, which_set)
   local inputs = data:narrow(2, 1, self._feature_size):clone()
   inputs = inputs:type('torch.DoubleTensor')
   inputs:resize(inputs:size(1), unpack(self._image_size))
   if self._scale then
      parent.rescale(inputs, self._scale[1], self._scale[2])
   end
   --inputs:resize(inputs:size(1), unpack(self._image_size))
   -- class 0 will have index 1, class 1 index 2, and so on.
   targets = targets:type('torch.DoubleTensor')
   -- construct inputs and targets dp.Views 
   local input_v, target_v = dp.ImageView(), dp.ClassView()
   input_v:forward(self._image_axes, inputs)
   target_v:forward('b', targets)
   target_v:setClasses(self._classes)
   -- construct dataset
   return dp.DataSet{inputs=input_v,targets=target_v,which_set=which_set}
end

function Insects:loadData()
   local path = os.getenv("DATA_PATH") .. "/insects/insects_" .. self._size .. "x" .. self._size .. ".t7"
   local data =  torch.load(path)
  return data
end

function Insects:loadTargets()
   local path = os.getenv("DATA_PATH") .. "/insects/".. self._labels_kind
   local targets, classes, str_classes = labels_to_indexes(path)
   return torch.Tensor(targets), classes, str_classes
end
