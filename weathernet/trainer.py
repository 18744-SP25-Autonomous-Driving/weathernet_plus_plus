import torch

def evaluate_loss(model, criterion, dataloader, device):
    # set model to eval mode
    model.eval()
    total_loss = 0.0
    # with torch.no_grad():
    #     for inputs, labels in dataloader:
    #         # load inputs and labels to device
    #         inputs, labels = inputs.to(device), labels.to(device)
    #         # Call model and get outputs
    #         outputs = model(inputs)
    #         # Calculate the loss using loss function
    #         loss = criterion(outputs, labels)
    #         total_loss += loss.item()
    # average_loss = total_loss / len(dataloader)

    with torch.no_grad():
        for inputs, labels in dataloader:
            # load inputs and labels to device
            inputs, labels = inputs.to(device=device), labels.to(device=device)
            # Separate labels for each pipeline
            fog_labels, glare_labels, weather_labels, night_labels = map_labels(labels, device)
            # FORWARD PASS: call model and get outputs
            outputs = model(inputs)
            (night_out, glare_out, weather_out, fog_out) = outputs
            # Calculate the loss using loss function
            loss = model.compute_loss(
                (night_out, glare_out, weather_out, fog_out),
                (night_labels, glare_labels, weather_labels, fog_labels),
            )
            total_loss += loss.item()
    average_loss = total_loss / len(dataloader)
    return average_loss


def evaluate_accuracy(model, dataloader, device):
    # set model to eval mode
    model.eval()

    correct = 0
    total = 0

    # with torch.no_grad():
    #     for inputs, labels in dataloader:
    #         # load inputs and labels to device
    #         inputs, labels = inputs.to(device), labels.to(device)
    #         # Call model and get outputs
    #         outputs = model(inputs)
    #         _, predicted = torch.max(outputs.data, 1)
    #         total += labels.size(0)
    #         correct += (predicted == labels).sum().item()

    # accuracy = (correct / total) * 100
    # return accuracy

    with torch.no_grad():
        for inputs, labels in dataloader:
            # load inputs and labels to device
            inputs, labels = inputs.to(device=device), labels.to(device=device)
            # Separate labels for each pipeline
            fog_labels, glare_labels, weather_labels, night_labels = map_labels(labels, device)
            # FORWARD PASS: call model and get outputs
            outputs = model(inputs)
            (night_out, glare_out, weather_out, fog_out) = outputs
            # Calculate accuracy
            _, night_predicted = night_out.max(1)
            _, weather_predicted = weather_out.max(1)
            total += (
                night_labels.size(0) + 
                weather_labels.size(0) + 
                glare_labels.size(0) + 
                fog_labels.size(0)
            )
            correct += (
                (night_predicted == night_labels).sum().item() +
                (weather_predicted == weather_labels).sum().item() + 
                (glare_out.squeeze() > 0.5).eq(glare_labels).sum().item() + 
                (fog_out.squeeze() > 0.5).eq(fog_labels).sum().item()
            )
    accuracy = (correct / total) * 100
    return accuracy

def map_labels(labels, device):
    """
    Separates labels ONLY into fog, glare, weather, and timeofday for WeatherNet
    """

    # TODO: Create enums for the pipeline categories, e.g. FOG = 0
    fog_labels = labels[:, 0]
    glare_labels = labels[:, 1]
    weather_labels = labels[:, 4]
    night_labels = labels[:, 6]
    # CSV category name is timeofday, but weathernet calls the pipeline nightnet

    return (
        fog_labels.to(device=device),
        glare_labels.to(device=device),
        weather_labels.to(device=device),
        night_labels.to(device=device),
    )

# TODO: generalize this function to work with any model. (or multiple models simultaneously)
# e.g. model = array of models, each trains using the same trainlaoder/testloader
# Separate WeatherNet into 4 separate ResNet50s?
def train(model, optimizer, criterion, trainloader, testloader, epochs, device):
    """
    Part 1.a: complete the training loop
    """
    train_losses = []  # For recording train losses
    test_losses = []  # For recording test losses

    # move model to device
    model.to(device=device)

    print("Begin training")
    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs}")
        running_loss = 0.0
        # Set the model to train mode
        model.train()

        # for inputs, labels in trainloader:
        batch_idx: int = 0
        num_batches = len(trainloader)
        for batch_idx, (inputs, labels) in enumerate(trainloader):
            if (batch_idx + 1) % 10 == 0 or batch_idx == 0:
                print(f"* Batch {batch_idx+1}/{num_batches}")
            optimizer.zero_grad()

            # load inputs and labels to device
            inputs, labels = inputs.to(device=device), labels.to(device=device)
            # Separate labels for each pipeline
            fog_labels, glare_labels, weather_labels, night_labels = map_labels(labels, device)
            # FORWARD PASS: call model and get outputs
            outputs = model(inputs)
            (night_out, glare_out, weather_out, fog_out) = outputs

            # Calculate the loss using loss function
            # loss = criterion(outputs, labels)
            loss = model.compute_loss(
                (night_out, glare_out, weather_out, fog_out),
                (night_labels, glare_labels, weather_labels, fog_labels),
            )
            # BACKWARDS PASS: call backward on loss
            running_loss += loss.item()
            loss.backward()
            # Add optimizer step
            optimizer.step()

        train_loss = running_loss / len(trainloader)
        train_losses.append(train_loss)
        print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.2f}")

        test_loss = evaluate_loss(model, criterion, testloader, device)
        test_losses.append(test_loss)
        print(f"Epoch {epoch+1}/{epochs} - Test Loss: {test_loss:.2f}")

        test_accuracy = evaluate_accuracy(model, testloader, device)
        print(f"Epoch {epoch+1}/{epochs} - Test Accuracy: {test_accuracy:.2f}%")

    return train_losses, test_losses