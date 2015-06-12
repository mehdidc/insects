
local CustomExperimentFactory = torch.class("dp.CustomExperimentFactory", "dp.ExperimentFactory")
CustomExperimentFactory.isCustomExperimentFactory = true

function CustomExperimentFactory:build(hyperparameters, experiment_id)
    return hyperparameters.builder(hyperparameters)
end

