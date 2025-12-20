class ModelConfiguration :
    """Class to manage and get model configuration parameters."""
    # Initializes the model configuration with given parameters
    def __init__(self, model_name: str, 
                 learning_rate: float, 
                 num_epochs: int):
        # assert the types of each parameter
        assert isinstance (model_name, str), "model_name must be a string"
        assert isinstance (learning_rate, float), "learning_rate must be a float"
        assert isinstance (num_epochs, int), "num_epochs must be an integer or None" 

        # assign parameters to instance variables
        self.model_name = model_name
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
    
    # prints a summary of the model configuration
    def print_summary(self) -> None:
        print(f"model name: {self.model_name} fitted for {self.num_epochs} epochs with a learning rate of {self.learning_rate}")
    
    # Adjust model learning rate
    def adjust_learning_rate(self, new_learning_rate: float) -> None:
        assert isinstance (new_learning_rate, float), "new_learning_rate must be a float"
        self.learning_rate = new_learning_rate

    # Adjust model number of epochs
    def adjust_num_epochs(self, new_num_epochs: int) -> None:
        assert isinstance (new_num_epochs, int), "new_num_epochs must be an integer or None"
        self.num_epochs = new_num_epochs

    # Retrieve model configuration as a dictionary
    def get_hyperparametres(self) -> dict:
        return {
            "model_name": self.model_name,
            "learning_rate": self.learning_rate,
            "num_epochs": self.num_epochs
        }

    

# test the ModelConfiguration class
if __name__ == "__main__":
    config_model_1 = ModelConfiguration("resnet50", 0.001, 25)
    config_model_1.print_summary()
    config_model_1.adjust_learning_rate(0.0005)
    config_model_1.adjust_num_epochs(30)            
    config_model_1.print_summary()
    print(config_model_1.get_hyperparametres())


    config_model_2 = ModelConfiguration("vgg16", 0.01, 30)
    config_model_2.print_summary()  
    config_model_2.adjust_learning_rate(0.005)
    config_model_2.adjust_num_epochs(50)        
    config_model_2.print_summary()
    print(config_model_2.get_hyperparametres())


