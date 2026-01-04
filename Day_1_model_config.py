class ModelConfiguration:
    """Class to manage and get model configuration parameters."""
    # Initializes the model configuration with given parameters
    def __init__(self, model_name: str, 
                 learning_rate: float, 
                 num_epochs: int):
        # assert the types of each parameter
        if not isinstance(model_name, str): 
            raise TypeError(f"model_name must be a string bit got:{type(model_name).__name__}")
        if not isinstance(learning_rate, float):
            raise TypeError(f"learning_rate must be a float:{type(learning_rate).__name__}")
        if not isinstance(num_epochs, int):
            raise TypeError("num_epochs must be an integer or None:{type(num_epochs).__name__}")

        # assign parameters to instance variables
        self.model_name = model_name
        self._learning_rate = learning_rate
        self.num_epochs = num_epochs
    
    # prints a summary of the model configuration
    def __str__(self):
        return f"model name: {self.model_name} fitted for {self.num_epochs} epochs with a learning rate of {self.learning_rate}"
        
    
    # Adjust model learning rate
    def adjust_learning_rate(self, new_learning_rate: float) -> None:
        if not isinstance (new_learning_rate, float):
            raise TypeError(f"new_learning_rate must be a float but got :{type(new_learning_rate).__name__}")
        self.learning_rate = new_learning_rate

    # Adjust model number of epochs
    def adjust_num_epochs(self, new_num_epochs: int) -> None:
        if not isinstance (new_num_epochs, int):
            raise TypeError(f"new_num_epochs must be an integer but got:{type(new_num_epochs).__name__}")
        self.num_epochs = new_num_epochs

    # Retrieve model configuration as a dictionary
    def get_hyperparameters(self) -> dict:
        return {
            "model_name": self.model_name,
            "learning_rate": self.learning_rate,
            "num_epochs": self.num_epochs
        }

    

# test the ModelConfiguration class
if __name__ == "__main__":
    config_model_1 = ModelConfiguration("resnet50", 0.001, 25)
    print(config_model_1.__str__())
    config_model_1.adjust_learning_rate(0.0005)
    config_model_1.adjust_num_epochs(30)            
    print(config_model_1.__str__())
    print(config_model_1.get_hyperparameters())


    config_model_2 = ModelConfiguration("vgg16", 0.01, 30)
    print(config_model_2.__str__())
    config_model_2.adjust_learning_rate(0.005)
    config_model_2.adjust_num_epochs(50)        
    config_model_2.__str__()
    print(config_model_2.get_hyperparameters())


