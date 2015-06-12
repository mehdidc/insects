
------------------------------------------------------------------------
----[[ DatasourceFactory ]]--
---- interface, factory
---- A datasource factory that can be used to build datasources given
---- a table of hyper-parameters
--------------------------------------------------------------------------
--
local CustomDatasourceFactory, parent = torch.class("dp.CustomDatasourceFactory", "dp.DatasourceFactory")
CustomDatasourceFactory.isCustomDatasourceFactory = true

function CustomDatasourceFactory:build(opt)
   return(opt.datasource)
end
