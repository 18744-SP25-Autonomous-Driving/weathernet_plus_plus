import torch
import os


def evaluate_loss(model, criterion, dataloader, device):
    # set model to eval mode
    model.eval()
    total_loss = 0.0
    total_losses = torch.zeros(model.get_num_pipelines()).to(device=device)
    with torch.no_grad():
        for inputs, labels in dataloader:
            # load inputs and labels to device
            inputs, labels = inputs.to(device=device), labels.to(device=device)

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

            # FORWARD PASS: call model and get outputs
            outputs = model(inputs)

            # Calculate accuracy
            if model.get_num_pipelines() == 4: # WeatherNet
                fog_out = outputs[:, 0]
                glare_out = outputs[:, 1]
                weather_out = outputs[:, 2:8]
                tod_out = outputs[:, 8:12]

                fog_labels = labels[:, 0].float() # float for binary classification tasks
                glare_labels = labels[:, 1].float()
                weather_labels = labels[:, 4]
                tod_labels = labels[:, 6]

                # convert raw logits on binary classifiers to categorical labels
                fog_out = fog_out.squeeze() > 0.5
                glare_out = glare_out.squeeze() > 0.5

                # convert raw logits on multiclass to categorical labels
                weather_out = torch.argmax(weather_out, dim=1)
                tod_out = torch.argmax(tod_out, dim=1)

                fog_eval = (fog_out == fog_labels).sum().item()
                glare_eval = (glare_out == glare_labels).sum().item()
                weather_eval = (weather_out == weather_labels).sum().item()
                tod_eval = (tod_out == tod_labels).sum().item()

                evals = torch.tensor([
                    fog_eval,
                    glare_eval,
                    weather_eval,
                    tod_eval
                ]).to(device=device)

                corrects += evals # element-wise sum individual pipelines
                total += (
                    fog_labels.size(0)
                    + glare_labels.size(0)
                    + weather_labels.size(0)
                    + tod_labels.size(0)
                )
                correct += (
                    (fog_out == fog_labels).sum().item()
                    + (glare_out == glare_labels).sum().item()
                    + (weather_out == weather_labels).sum().item()
                    + (tod_out == tod_labels).sum().item()
                )
            elif model.get_num_pipelines() == 7: # WeatherNet++, MtlWeatherNet, WeatherNetTransformer
                fog_out = outputs[:, 0]
                glare_out = outputs[:, 1]
                road_out = outputs[:, 2:5]
                traffic_out = outputs[:, 5:8]
                weather_out = outputs[:, 8:14]
                scene_out = outputs[:, 14:18]
                tod_out = outputs[:, 18:22]

                fog_labels = labels[:, 0].float() # float for binary classification tasks
                glare_labels = labels[:, 1].float()
                road_labels = labels[:, 2]
                traffic_labels = labels[:, 3]
                weather_labels = labels[:, 4]
                scene_labels = labels[:, 5]
                tod_labels = labels[:, 6]

                # convert raw logits on binary classifiers to categorical labels
                fog_out = fog_out.squeeze() > 0.5
                glare_out = glare_out.squeeze() > 0.5

                # convert raw logits on multiclass to categorical labels
                road_out = torch.argmax(road_out, dim=1)
                traffic_out = torch.argmax(traffic_out, dim=1)
                weather_out = torch.argmax(weather_out, dim=1)
                scene_out = torch.argmax(scene_out, dim=1)
                tod_out = torch.argmax(tod_out, dim=1)

                fog_eval = (fog_out == fog_labels).sum().item()
                glare_eval = (glare_out == glare_labels).sum().item()
                road_eval = (road_out == road_labels).sum().item()
                traffic_eval = (traffic_out == traffic_labels).sum().item()
                weather_eval = (weather_out == weather_labels).sum().item()
                scene_eval = (scene_out == scene_labels).sum().item()
                tod_eval = (tod_out == tod_labels).sum().item()

                evals = torch.tensor([
                    fog_eval,
                    glare_eval,
                    road_eval,
                    traffic_eval,
                    weather_eval,
                    scene_eval,
                    tod_eval
                ]).to(device=device)

                corrects += evals # element-wise sum individual pipelines
                total += (
                    fog_labels.size(0)
                    + glare_labels.size(0)
                    + road_labels.size(0)
                    + traffic_labels.size(0)
                    + weather_labels.size(0)
                    + scene_labels.size(0)
                    + tod_labels.size(0)
                )
                correct += (
                    (fog_out == fog_labels).sum().item()
                    + (glare_out == glare_labels).sum().item()
                    + (road_out == road_labels).sum().item()
                    + (traffic_out == traffic_labels).sum().item()
                    + (weather_out == weather_labels).sum().item()
                    + (scene_out == scene_labels).sum().item()
                    + (tod_out == tod_labels).sum().item()
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
    tod_labels = labels[:, 6]
    # CSV category name is timeofday, but weathernet calls the pipeline todnet

    return (
        fog_labels.to(device=device),
        glare_labels.to(device=device),
        weather_labels.to(device=device),
        tod_labels.to(device=device),
    )


# TODO: generalize this function to work with any model. (or multiple models simultaneously)
# e.g. model = array of models, each trains using the same trainlaoder/valloader
# Separate WeatherNet into 4 separate ResNet50s?
def train(model, optimizer, criterion, trainloader, valloader, epochs, device, data_save_dir, writer=None):
    """
    Part 1.a: complete the training loop
    """
    # set up running losses for logging
    num_pipelines = model.get_num_pipelines()
    running_losses = torch.zeros(num_pipelines).to(device=device)

    if num_pipelines == 4:
        pipelines = ["fog", "glare", "weather", "tod"]
    else:
        pipelines = ["fog", "glare", "road", "traffic", "weather", "scene", "tod"]

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
        # Separate labels for each pipeline
        model_dim = model.get_num_pipelines()
        for batch_idx, (inputs, labels) in enumerate(trainloader):
            optimizer.zero_grad()

            # load inputs and labels to device
            inputs, labels = inputs.to(device=device), labels.to(device=device)

            # FORWARD PASS: call model and get outputs
            outputs = model(inputs)

            # Calculate the loss using loss function
            # 'losses' is a tensor of loss tensors from each pipeline
            # e.g. [loss_fog, loss_glare, loss_road, loss_traffic, loss_weather, loss_scene, loss_tod]
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
        losses_msg = " | ".join(f"{pl} loss: {loss:.2f}" for pl, loss in zip(pipelines, train_losses))
        print(f"Epoch {epoch+1}/{epochs} - [TRAIN] - {losses_msg}")

        val_loss, val_losses = evaluate_loss(model, criterion, valloader, device)
        losses_msg = " | ".join(f"{pl} loss: {loss:.2f}" for pl, loss in zip(pipelines, val_losses))
        print(f"Epoch {epoch+1}/{epochs} - [VAL] - {losses_msg}")

        val_accuracy, val_accuracies = evaluate_accuracy(model, valloader, device)
        accs_msg = " | ".join(f"{pl} accuracy: {acc:.2f}%" for pl, acc in zip(pipelines, val_accuracies))
        print(f"Epoch {epoch+1}/{epochs} - [VAL] - {accs_msg}")

        # Ensure the data save directory exists
        os.makedirs(data_save_dir, exist_ok=True)
        # Construct the filename for the checkpoint
        model_name = model.get_name()
        checkpoint_filename = os.path.join(data_save_dir, f"{model_name}_epoch_{epoch+1}.pth")
        # Save the model checkpoint
        model.save_checkpoint(checkpoint_filename)
        print(f"Checkpoint saved: {checkpoint_filename}")

        # log to tensorboard summarywriter
        if writer:
            # per-pipeline training losses/val accuracies
            for i, pl in enumerate(pipelines):
                writer.add_scalar(f"Loss/train/{pl}", train_losses[i], epoch)
                writer.add_scalar(f"Loss/val/{pl}", train_losses[i], epoch)
                writer.add_scalar(f"Accuracy/val/{pl}", val_accuracies[i], epoch)
            writer.add_scalar("Loss/train", train_loss, epoch)
            writer.add_scalar("Loss/val", val_loss, epoch)
            writer.add_scalar("Accuracy/val", val_accuracy, epoch)
