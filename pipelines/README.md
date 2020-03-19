# Pipelines

This is an example of setting up pipelines to work with the policy in this
repository.

## Install Dependencies

Before proceeding with this tutorial, you'll need to install OpenShift
Pipelines by installing the OpenShift Pipelines Operator:
1. You can view the instructions to install the OpenShift Pipelines Operator
   [here](https://github.com/openshift/tektoncd-pipeline-operator) or if you
   have an OpenShift cluster running, navigate to the Operators drop-down menu
   on the left and click on "OperatorHub". Then search for OpenShift Pipelines
   Operator and install it.
1. Install OPA Gatekeeper by following these
   [instructions](https://github.com/open-policy-agent/gatekeeper#installation).
   This tutorial was constructed using OPA Gatekeeper `v3.1.0-beta.7`. Once
   installed, make sure to add the `anyuid` security context constraint to the
   `gatekeeper-admin` service account by running:
   ```bash
   oc adm policy add-scc-to-user anyuid -z gatekeeper-admin
   ```
1. Install [podman](https://podman.io/).

## Fork This Repository

You'll want to fork this repository in order run through the tutorial so that
you can commit and push changes to trigger builds.

## Configure the cluster

- Create and set the Namespace where the resoures will live by using the
  `new-project` sub-command:

```bash
oc new-project policy-pipeline
```

- Create the trigger admin service account, role and rolebinding

```bash
oc apply -f ./pipelines/admin-role.yaml
```

- Create the webhook user, role and rolebinding

```bash
oc apply -f ./pipelines/webhook-role.yaml
```

## Install the Pipeline and Trigger

### Install the Pipeline

To install the pipeline run:

```bash
oc apply -f ./pipelines/pipeline.yaml
```

### Install the TriggerTemplate, TriggerBinding and EventListener

To install these run:

```bash
oc apply -f ./pipelines/triggers.yaml
```

## Add Ingress and GitHub-Webhook Tasks

```bash
oc apply -f ./pipelines/create-ingress.yaml
oc apply -f ./pipelines/create-webhook.yaml
```

## Run Ingress Task

Be sure to replace the `ExternalDomain` parameter value with your FQDN. This
will be used by the GitHub webhook to reach the ingress in your cluster in
order to pass the relevent GitHub commit details to the `EventListener` service
running in your cluster. Then run:

```bash
oc apply -f ./pipelines/ingress-run.yaml
```

## Run GitHub Webhook Task

You will need to create a [GitHub Personal Access
Token](https://help.github.com/en/articles/creating-a-personal-access-token-for-the-command-line#creating-a-token)
with the following access:

- public_repo
- admin:repo_hook

Next, create a secret like so with your access token.

```yaml
apiVersion: v1
kind: Secret
metadata:
  name: webhook-secret
    namespace: policy-pipeline
    stringData:
      token: YOUR-GITHUB-ACCESS-TOKEN
        secret: random-string-data
```

Next you'll want to edit the `webhook-run.yaml` file:
- Modify the `GitHubOrg` and `GitHubUser` fields to match your setup.
- Modify the `ExternalDomain` field to match the FQDN used in
  `ingress-run.yaml` for configuring the GitHub webhook to use this FQDN to
  talk to the `EventListener`.

Then Create the webhook task:

```bash
oc apply -f ./pipelines/webhook-run.yaml
```

## Watch the Trigger and Pipeline Work!

Commit and push an empty commit to your development repo.

```bash
git commit -a -m "build commit" --allow-empty && git push origin mybranch
```

## Cleanup

Delete the namespaces:

```bash
oc delete namespace policy-pipeline
```
