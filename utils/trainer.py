import torch


def evaluate_loss(model, criterion, dataloader, device):
    # set model to eval mode
    model.eval()
    total_loss = 0.0
    total_losses = torch.zeros(model.get_num_pipelines()).to(device=device)
    with torch.no_grad():
        for inputs, labels in dataloader:
            # load inputs and labels to device
            inputs, labels = inputs.to(device=device), labels.to(device=device)

            # Separate labels for each pipeline
            model_dim = model.get_num_pipelines()
            labels = labels[:, :model_dim]

            # FORWARD PASS: call model and get outputs
            outputs = model(inputs)
            # Calculate the loss using loss function
            loss, losses = model.compute_loss(
                outputs,
                labels,
            )
            total_loss += loss.item()
            total_losses += losses # element-wise sum individual pipeline losses

    average_loss = total_loss / len(dataloader)
    average_losses = total_losses / len(dataloader)
    return average_loss, average_losses


def evaluate_accuracy(model, dataloader, device):
    # set model to eval mode
    model.eval()

    correct = 0
    corrects = torch.zeros(model.get_num_pipelines()).to(device=device) # per pipeline tracking
    total = 0

    with torch.no_grad():
        for inputs, labels in dataloader:
            # load inputs and labels to device
            inputs, labels = inputs.to(device=device), labels.to(device=device)

            # Separate labels for each pipeline
            model_dim = model.get_num_pipelines()
            labels = labels[:, :model_dim]

            # FORWARD PASS: call model and get outputs
            outputs = model(inputs)

            # TODO: Make this generic to support any of our models (ie. 7 pipelines and 4 pipelines)
            night_out = outputs[:, 0:4]
            glare_out = outputs[:, 4]
            weather_out = outputs[:, 5:11]
            fog_out = outputs[:, 11]

            # convert raw logits on multiclass to categorical labels
            night_out = torch.argmax(night_out, dim=1)
            weather_out = torch.argmax(weather_out, dim=1)

            # convert raw logits on binary classifiers to categorical labels
            glare_out = glare_out.squeeze() > 0.5
            fog_out = fog_out.squeeze() > 0.5

            # Parse labels
            night_labels = labels[:, 0]
            glare_labels = labels[:, 1].float() # float for binary classification tasks
            weather_labels = labels[:, 2]
            fog_labels = labels[:, 3].float()

            night_eval = (night_out == night_labels).sum().item()
            glare_eval = (glare_out == glare_labels).sum().item()
            weather_eval = (weather_out == weather_labels).sum().item()
            fog_eval = (fog_out == fog_labels).sum().item()

            # Calculate accuracy
            if model.get_num_pipelines() == 4:
                evals = torch.tensor([night_eval, glare_eval, weather_eval, fog_eval]).to(device=device)
                corrects += evals # element-wise sum individual pipelines
                # TODO: clean this up. e.g. num_pipelines * len(dataloader.dataset)?
                total += (
                    night_labels.size(0)
                    + weather_labels.size(0)
                    + glare_labels.size(0)
                    + fog_labels.size(0)
                )
                correct += (
                    (night_out == night_labels).sum().item()
                    + (weather_out == weather_labels).sum().item()
                    + (glare_out == glare_labels).sum().item()
                    + (fog_out == fog_labels).sum().item()
                )
            elif model.get_num_pipelines() == 7:
                road_out = outputs[:, 12:15]
                traffic_out = outputs[:, 15:18]
                scene_out = outputs[:, 18:22]

                road_labels = labels[:, 4]
                traffic_labels = labels[:, 5]
                scene_labels = labels[:, 6]

                # convert raw logits on multiclass to categorical labels
                road_out = torch.argmax(road_out, dim=1)
                traffic_out = torch.argmax(traffic_out, dim=1)
                scene_out = torch.argmax(scene_out, dim=1)

                road_eval = (road_out == road_labels).sum().item()
                traffic_eval = (traffic_out == traffic_labels).sum().item()
                scene_eval = (scene_out == scene_labels).sum().item()

                evals = torch.tensor([night_eval, glare_eval, weather_eval, fog_eval, road_eval, traffic_eval, scene_eval]).to(device=device)
                corrects += evals # element-wise sum individual pipelines
                total += (
                    night_labels.size(0)
                    + glare_labels.size(0)
                    + weather_labels.size(0)
                    + fog_labels.size(0)
                    + road_labels.size(0)
                    + traffic_labels.size(0)
                    + scene_labels.size(0)
                )
                correct += (
                    (night_out == night_labels).sum().item()
                    + (weather_out == weather_labels).sum().item()
                    + (glare_out == glare_labels).sum().item()
                    + (fog_out == fog_labels).sum().item()
                    + (road_out == road_labels).sum().item()
                    + (traffic_out == traffic_labels).sum().item()
                    + (scene_out == scene_labels).sum().item()
                )

    accuracy = (correct / total) * 100
    accuracies = (corrects / len(dataloader.dataset)) * 100 # element-wise accuracy for each pipeline
    return accuracy, accuracies


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
def train(model, optimizer, criterion, trainloader, testloader, epochs, device, writer=None):
    """
    Part 1.a: complete the training loop
    """
    train_loss_log = []  # For logging train losses
    test_loss_log = []  # For logging test losses

    # set up running losses for logging
    running_losses = torch.zeros(model.get_num_pipelines()).to(device=device)

    # first four: base WeatherNet
    # all seven: WeatherNet++, MTL, Transformer models
    all_pipelines = ["night", "glare", "weather", "fog", "road", "traffic", "scene"]
    pipelines = all_pipelines[:model.get_num_pipelines()]

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
            optimizer.zero_grad()

            # load inputs and labels to device
            inputs, labels = inputs.to(device=device), labels.to(device=device)
            # Separate labels for each pipeline
            model_dim = model.get_num_pipelines()

            # truncate labels to match model_dim
            labels = labels[:, :model_dim]
            # FORWARD PASS: call model and get outputs
            outputs = model(inputs)
            # print(outputs)

            # Calculate the loss using loss function
            # loss = criterion(outputs, labels)
            # losses is a tensor of loss tensors from each pipeline
            # [loss_night, loss_glare, loss_weather, loss_fog]
            loss, losses = model.compute_loss(
                outputs,
                labels,
            )
            # BACKWARDS PASS: call backward on loss
            # NOTE: currently doing backprop on the SUM of all losses
            running_loss += loss.item()
            running_losses += losses # element-wise sum individual pipeline losses
            loss.backward()
            # Add optimizer step
            optimizer.step()

            # LOG MESSAGE: individual pipeline losses
            losses_msg = " | ".join(f"{pl} loss: {loss:.2f}" for pl, loss in zip(pipelines, losses))

            # Print the loss every 10 batches
            if (batch_idx + 1) % 10 == 0:
                print(f"* Batch {batch_idx+1}/{num_batches} - {losses_msg}")

        train_loss = running_loss / len(trainloader)
        train_losses = running_losses / len(trainloader)
        train_loss_log.append(train_loss)
        losses_msg = " | ".join(f"{pl} loss: {loss:.2f}" for pl, loss in zip(pipelines, train_losses))
        print(f"Epoch {epoch+1}/{epochs} - [TRAIN] - {losses_msg}")

        test_loss, test_losses = evaluate_loss(model, criterion, testloader, device)
        test_loss_log.append(test_loss)
        losses_msg = " | ".join(f"{pl} loss: {loss:.2f}" for pl, loss in zip(pipelines, test_losses))
        print(f"Epoch {epoch+1}/{epochs} - [TEST] - {losses_msg}")

        test_accuracy, test_accuracies = evaluate_accuracy(model, testloader, device)
        accs_msg = " | ".join(f"{pl} accuracy: {acc:.2f}%" for pl, acc in zip(pipelines, test_accuracies))
        print(f"Epoch {epoch+1}/{epochs} - [TEST] - {accs_msg}")

        # log to tensorboard summarywriter
        if writer:
            # per-pipeline training losses/test accuracies
            for i, pl in enumerate(pipelines):
                writer.add_scalar(f"Loss/train/{pl}", train_losses[i], epoch)
                writer.add_scalar(f"Loss/test/{pl}", train_losses[i], epoch)
                writer.add_scalar(f"Accuracy/test/{pl}", test_accuracies[i], epoch)

            writer.add_scalar("Loss/train", train_loss, epoch)
            writer.add_scalar("Loss/test", test_loss, epoch)
            writer.add_scalar("Accuracy/test", test_accuracy, epoch)

    return train_loss_log, test_loss_log
