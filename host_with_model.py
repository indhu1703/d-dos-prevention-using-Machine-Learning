from http.server import SimpleHTTPRequestHandler, HTTPServer
import os
import json
import json
import numpy as np

class MyHTTPRequestHandler(SimpleHTTPRequestHandler):
    blocked_ips = {'192.168.149.172', '192.168.1.100'} 
    def do_GET(self):
        if self.client_address[0] in self.blocked_ips:
            self.send_error(403, "Forbidden")
            return
        
        if self.path == '/':
            self.path = 'index.html'  
        return SimpleHTTPRequestHandler.do_GET(self)

def run(server_class=HTTPServer, handler_class=MyHTTPRequestHandler, port=5000, bind_address=''):
    server_address = (bind_address, port)
    httpd = server_class(server_address, handler_class)
    print(f'Starting server on {bind_address}:{port}...')
    httpd.serve_forever()


def model():
    class neuralnet(nn.Module):
        nn = []
        def __init__(self,input_size,hidden_size,num_classes):
            super(neuralnet, self).__init__()
            self.l1 = nn.Linear(input_size, hidden_size)
            self.l2 = nn.Linear(hidden_size, hidden_size)
            self.l3 = nn.Linear(hidden_size, num_classes)
            self.relu = nn.ReLU()

        def forward(self, x):
            out = self.l1(x)
            out = self.relu(out)
            out = self.l2(out)
            out = self.relu(out)
            out = self.l3(out)
            return out
        
    class nlpDataset(Dataset):
        def __init__(self):
            self.n_samples = len(X_train)
            self.x_data = X_train
            self.y_data = Y_train

        def __getitem__(self,index):
            return self.x_data[index],self.y_data[index]

        def __len__(self):
            return self.n_samples
        
    
    with open("D:\\mini_project\\model_creating\\json_dataset.json",'r') as file:
        intents = json.load(file)

    all_words = []
    tags = []
    xy = []
    x_train = []
    y_train = []
    ignore_words = ['!','.',',','?',"'"]

    for intent in intents['intents']:
        tag = intent['description']
        tags.append(tag)
        for pattern in intent['patterns']:
            pass
    all_words = sorted(set(all_words))

    for (sentence, tag) in xy:

        label = tags.index(tag)
        y_train.append(label)

    X_train = np.array(x_train)
    Y_train = np.array(y_train)
    print('x_train,y_train completed')

    dataset = nlpDataset()
    input_size = len(X_train[0])
    output_size = len(tags)
    batch_size = 8
    hidden_size = 8
    learning_rate = 0.001
    num_epochs = 150
    print('hyperparameter defined ')
    train_loader = DataLoader(dataset=dataset, batch_size=batch_size,shuffle=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = neuralnet(input_size,hidden_size,output_size).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    print('epoch started')
    for epoch in range(num_epochs):
        for (words,labels) in train_loader:
            words = words.to(device)
            labels = labels.to(dtype = torch.long).to(device)
            output = model(words)
            loss = criterion(output,labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


        if epoch % 10 == 0 :
            print(f'epoch {epoch+1}/{num_epochs}, loss = {loss.item():.4f}')

    print(f'final loss = {loss.item():.4f}')



    data = {"model_state":model.state_dict(),
            "input_size":input_size,
            "output_size":output_size,
            "hidden_size":hidden_size,
            "all_words":all_words,
            "tags":tags
            }

    File = "analyse_text.pth"
    torch.save(data, File)
    print("model saved")
    model.eval()

    FILE = 'D:\\mini_project\\model_creating\\analyse_text.pth'
    data = torch.load(FILE)

    input_size = data["input_size"]
    hidden_size = data["hidden_size"]
    output_size = data["output_size"]
    all_words = data['all_words']
    tags = data['tags']
    model_state = data["model_state"]

    model = neuralnet(input_size, hidden_size, output_size).to(device)
    model.load_state_dict(model_state)
    model.eval()


    assistant_name = "nancy"
    print("started")

    while True:
        sentence = input("you : ")

        for i,word in enumerate(sentence):
            sentence[i] = []
        print(sentence)
        X = sentence
        X = 1
        X = torch.from_numpy(X)

        output = model(X)

        _, predicted = torch.max(output,dim=1)
        prob_softmax = torch.softmax(output, dim=1)
        prob = prob_softmax[0][predicted.item()]
        print('final prob = ',prob.item())
        tag = tags[predicted.item()]
        print(tag)
        for intent in intents['intents']:
            if tag == intent['description']:
                print(intent['description'])
                if prob.item() >= 0.99:
                    print("command found")
                else:
                    print("can't find command")

if __name__ == "__main__":
    
    bind_address = '192.168.149.6'  
    run(bind_address=bind_address)