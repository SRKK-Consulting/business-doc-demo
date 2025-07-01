# Business Doc Demo

This project is a web application deployed using Docker and Azure Container Apps. The application is containerized, pushed to Docker Hub, and hosted on Azure Container Apps for scalable deployment.

## Prerequisites

Before you begin, ensure you have the following installed and configured:

- **Docker**: Install [Docker Desktop](https://www.docker.com/products/docker-desktop/) to build and push container images.
- **Docker Hub Account**: Create an account on [Docker Hub](https://hub.docker.com/) to store your images.
- **Azure CLI**: Install the [Azure CLI](https://docs.microsoft.com/en-us/cli/azure/install-azure-cli) and log in with `az login`.
- **Azure Subscription**: Ensure you have an active Azure subscription and a resource group named `Apps`.
- **Azure Container Apps Environment**: A Container Apps environment must be set up in your resource group.

## Project Setup

1. **Clone the Repository**
   ```bash
   git clone <repository-url>
   cd business-doc-demo
   ```

2. **Directory Structure**
   Ensure your project contains a `Dockerfile` for building the container image. A typical structure might look like:
   ```
   .
   ├── Dockerfile
   ├── app.py (or your main application file)
   ├── requirements.txt (if using Python)
   └── ...
   ```

## Deployment Steps

### Step 1: Build and Push Docker Image to Docker Hub

1. **Log in to Docker Hub**
   Authenticate with Docker Hub using your credentials:
   ```bash
   docker login
   ```
   Enter your Docker Hub username and password when prompted.

2. **Build the Docker Image**
   Build the Docker image locally, tagging it with your Docker Hub username and repository:
   ```bash
   docker build -t sohzongxian/business-doc-demo:latest .
   ```

3. **Push the Image to Docker Hub**
   Push the built image to your Docker Hub repository:
   ```bash
   docker push sohzongxian/business-doc-demo:latest
   ```

4. **Test the Image Locally (Optional)**
   Run the container locally to verify it works:
   ```bash
   docker run -p 5000:5000 sohzongxian/business-doc-demo:latest
   ```
   Access the application at `http://localhost:5000` in your browser.

### Step 2: Deploy to Azure Container Apps

1. **Set Up Azure Container Apps Environment**
   If you haven't already created a Container Apps environment, create one in your resource group:
   ```bash
   az containerapp env create \
     --name <environment-name> \
     --resource-group Apps \
     --location <azure-region>
   ```
   Replace `<environment-name>` with your desired environment name (e.g., `myContainerAppEnv`) and `<azure-region>` with a supported Azure region (e.g., `eastus`).

2. **Deploy the Application**
   Deploy the Docker Hub image to Azure Container Apps:
   ```bash
   az containerapp up \
     --name business-doc-demo \
     --resource-group Apps \
     --environment <environment-name> \
     --image docker.io/sohzongxian/business-doc-demo:latest \
     --ingress external \
     --target-port 5000
   ```
   - This command creates or updates a Container App named `business-doc-demo`.
   - The `--ingress external` flag makes the app publicly accessible.
   - The `--target-port 5000` specifies the port your application listens on.

3. **Update the Application (If Needed)**
   If you update the Docker image and push a new version, update the Container App:
   ```bash
   az containerapp update \
     --name business-doc-demo \
     --resource-group Apps \
     --image docker.io/sohzongxian/business-doc-demo:latest
   ```

### Step 3: Verify Deployment

- After deployment, Azure Container Apps will provide a public URL for your application (e.g., `https://business-doc-demo.<region>.azurecontainerapps.io`).
- Access the URL in a browser to confirm the application is running.
- Use the Azure Portal or CLI to monitor logs and metrics:
  ```bash
  az containerapp logs show \
    --name business-doc-demo \
    --resource-group Apps
  ```

## Troubleshooting

- **Docker Push Fails**: Ensure you are logged in with `docker login` and have the correct Docker Hub credentials.
- **Azure Deployment Fails**: Verify that the resource group `Apps` and the Container Apps environment exist. Check the image tag and ensure the image is accessible on Docker Hub.
- **Application Not Accessible**: Confirm that the `--target-port` matches the port your application exposes (5000 in this case) and that `--ingress external` is set.
- **Authentication Errors**: For private Docker Hub images, ensure you provide valid credentials using `--registry-username` and `--registry-password` in the `az containerapp up` command.

## Additional Notes

- The application is exposed on port `5000`. Update the `--target-port` if your application uses a different port.
- For private Docker Hub images, add the following flags to the `az containerapp up` command:
  ```bash
  --registry-server hub.docker.com \
  --registry-username sohzongxian \
  --registry-password <your-docker-hub-password>
  ```
- To scale the application, use the Azure Portal or CLI to adjust the Container App's scaling settings.

## License

This project is licensed under the MIT License.