class Discriminator(nn.Module):
    """Predicts the skill being used from the state
    state_dim is the dimension of the state space
    num_skills is the number of skills to predict
    Remember to use only state_dim for the discriminator not state with skill
    """
    def __init__(self, state_dim: int=512, num_skills:int=4):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, num_skills) #output is logits per skill
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        logits =  self.fc3(x)  #no softmax use with XELoss
        return logits

#Train discriminator every rollout


