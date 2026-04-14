from kubernetes import client, config

def scale_deployment(replicas: int, namespace: str = "default"):
    try:
        # Try in-cluster config first (running inside a pod),
        # fall back to local kubeconfig for development.
        try:
            config.load_incluster_config()
        except config.ConfigException:
            config.load_kube_config()

        apps_v1 = client.AppsV1Api()
        deployment_name = "5g-traffic-service"

        body = {"spec": {"replicas": replicas}}
        apps_v1.patch_namespaced_deployment_scale(
            name=deployment_name,
            namespace=namespace,
            body=body,
        )
        print(f"🚀 Scaled '{deployment_name}' to {replicas} replicas.")

    except client.exceptions.ApiException as api_err:
        print(f"❌ Kubernetes API error: {api_err}")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")