from pathlib import Path


SRC = Path("/Users/fengxun/Downloads/main.tex")
OUT = Path("artifacts/main_reviewer_revised.tex")


def replace_once(text, old, new):
    if old not in text:
        raise SystemExit(f"Missing expected block:\n{old[:200]}...")
    return text.replace(old, new, 1)


def main():
    text = SRC.read_text(encoding="utf-8")

    text = replace_once(
        text,
        """Against this background, this study investigates whether an MM-VAE combining VAE-based latent-space regularization with a supervised classification objective can outperform unimodal and late-fusion baselines, and whether it learns genuine cross-modal complementarity beyond reliance on a single dominant feature such as HGB.

This study proposes a patient-level anemia screening framework based on MM-VAE, which fuses CBC structured indicators and RBC microscopy image embeddings into a shared latent space via a PoE mechanism, trained end-to-end under a joint objective combining reconstruction and classification supervision.

This study designs a mask-based CBC reconstruction loss and a Kullback--Leibler (KL) warm-up training strategy. The PoE fusion mechanism naturally supports missing-modality inference without retraining separate models for different missing-data scenarios.

This study systematically verifies the model's cross-modal complementarity and inference robustness through an HGB ablation experiment and a missing-modality experiment, and quantifies the stability advantage of MM-VAE over Concat Fusion via multi-seed evaluation.""",
        """Against this background, this study investigates whether an MM-VAE combining VAE-based latent-space regularization with a supervised classification objective can outperform unimodal and direct fusion baselines, and whether it learns genuine cross-modal complementarity beyond reliance on a single dominant feature such as HGB. In addition to simple feature concatenation, recent multimodal learning studies have also explored more expressive fusion paradigms such as attention-based fusion and cross-modal Transformer architectures. While these methods can model inter-modal interactions, they are usually more dependent on larger sample sizes, higher model capacity, and more complete paired inputs, which may not be ideal for a relatively small medical dataset with potential missing-modality scenarios. By contrast, the MM-VAE framework provides a probabilistic latent-space formulation that is better suited to heterogeneous medical evidence and more naturally accommodates incomplete inputs.

This study proposes a patient-level anemia screening framework based on MM-VAE, which fuses CBC structured indicators and RBC microscopy image embeddings into a shared latent space via a PoE mechanism, trained end-to-end under a joint objective combining reconstruction and classification supervision. Different from direct feature concatenation at the representation level, the PoE mechanism combines modality-specific posterior distributions at the probabilistic level, encouraging the shared latent representation to retain information that is jointly supported across modalities. In the anemia screening task, this is particularly meaningful because CBC indicators provide quantitative hematological evidence, whereas blood smear images provide morphological evidence; the two modalities are complementary but not equally strong or equally complete in all cases.

This study designs a mask-based CBC reconstruction loss and a Kullback--Leibler (KL) warm-up training strategy. The PoE fusion mechanism naturally supports missing-modality inference without retraining separate models for different missing-data scenarios. Compared with heavier cross-modal interaction models, this design also offers a relatively compact and stable solution for patient-level multimodal screening under limited data conditions.

This study systematically verifies the model's cross-modal complementarity and inference robustness through an HGB ablation experiment and a missing-modality experiment, and quantifies the stability advantage of MM-VAE over Concat Fusion via multi-seed evaluation.""",
    )

    text = replace_once(
        text,
        """Existing multimodal fusion methods usually include early fusion, late fusion and shared latent representation learning. The first two implementations are relatively simple, but the deep relationship between CBC numerical features and blood smear image features may not be fully modeled. In contrast, multimodal representation learning based on VAE can integrate different modes in the latent space and reflect the degree of deviation of the sample from the normal distribution by reconstructing the abnormal score formed by the error and KL dispersion. Although multimodal VAE has shown potential in multi-source data modeling, it still lacks full exploration in the anemia screening task of fusing CBC reports and red blood cell microscopy images.""",
        """Existing multimodal fusion methods usually include early fusion, late fusion, attention-based fusion, cross-modal interaction models, and shared latent representation learning. Early fusion and late fusion are relatively simple to implement, but they may not fully model the heterogeneous yet complementary relationship between CBC numerical features and blood smear image features. Attention-based fusion and cross-modal Transformer methods can capture richer inter-modal dependencies, but they are often more data-hungry, computationally heavier, and more reliant on sufficiently complete paired samples, which may reduce their practicality in small-scale medical datasets and missing-modality settings.

In contrast, multimodal representation learning based on VAE provides a probabilistic latent-space framework for integrating heterogeneous modalities. In particular, the PoE mechanism combines modality-specific posterior distributions so that the shared latent variables are encouraged to preserve consistent evidence across modalities while still allowing inference when one modality is absent. For anemia screening, where CBC reports provide quantitative hematological measurements and red blood cell microscopy provides visual morphological clues, such a probabilistic evidence-fusion mechanism is especially suitable. Although multimodal VAE has shown potential in multi-source medical data modeling, it still lacks sufficient exploration in the anemia screening task of jointly fusing CBC reports and red blood cell microscopy images.""",
    )

    text = replace_once(
        text,
        """This study adopts patient-level modeling. Each sample matches one original case, including one CBC report, twelve RBC patch images from the same case, and a binary label. To ensure independence across the experimental division, the training, validation, and test sets are determined before patch images are loaded. Therefore, all patches from the same case fall into a single subset with none crossing into another. The project uses stratified sampling, splitting the data into training, validation, and test sets at a 70/15/15 ratio.

For the CBC modality, structured indicators are parsed into a patient-level feature table. Two configurations are used: core\\_with\\_hgb and core\\_without\\_hgb. Compared to the former one, the latter drops HGB to test whether the model over-relies on it. Each feature also carries a missingness flag, 1 if observed and 0 if missing, forming the cbc\\_mask vector. Missing values are imputed with the training-set median. CBC features are then standardized using the training set's mean and standard deviation, and the same transform is applied to validation and test sets.

For image modality, raw patches are not fed directly into training. A frozen Residual Network with 18 layers is used to extract patch features first, then the twelve patches per case are mean-pooled into a single patient-level image embedding. While freezing the backbone and mean pooling sacrifices some discriminative capacity, this design is motivated by the small dataset size, which makes end-to-end fine-tuning prone to overfitting, and by the need to isolate fusion mechanism differences across experiments rather than conflating them with encoder-level variation.""",
        """This study adopts patient-level modeling. Each sample matches one original case, including one CBC report, twelve RBC patch images from the same case, and a binary label. To ensure independence across the experimental division, the training, validation, and test sets are determined before patch images are loaded. Therefore, all patches from the same case fall into a single subset with none crossing into another. The project uses stratified sampling, splitting the data into training, validation, and test sets at a 70/15/15 ratio, corresponding to 700 training cases, 150 validation cases, and 150 test cases.

For the CBC modality, structured indicators are parsed into a patient-level feature table. Two core configurations are used in this study. The core\\_with\\_hgb setting contains RBC, HGB, HCT, MCV, MCH, MCHC, and RDW-CV, while the core\\_without\\_hgb setting removes HGB and keeps RBC, HCT, MCV, MCH, MCHC, and RDW-CV. The latter is used to test whether the model over-relies on HGB. For each CBC indicator, a missingness flag is also generated, where 1 indicates that the value is observed and 0 indicates that the value is missing, forming the cbc\\_mask vector. Missing CBC values are imputed using the median value calculated from the training set only. After imputation, CBC features are standardized by z-score normalization using the mean and standard deviation of the training set, and the same transformation is then applied to the validation and test sets. This avoids information leakage from validation or test samples into the preprocessing procedure.

For the image modality, raw patches are not directly used in the final MM-VAE training stage. Each case contains twelve RBC patch images. During feature extraction, each patch is resized to 224 $\\times$ 224 and normalized using the standard RGB mean and standard deviation values of ImageNet. A frozen ResNet-18 backbone with ImageNet-pretrained weights is used as the patch-level image encoder. The final classification layer of ResNet-18 is removed, and the resulting 512-dimensional feature vector is used as the patch embedding. The twelve patch embeddings from the same case are then mean-pooled into a single 512-dimensional patient-level image embedding. This cached embedding design reduces repeated CNN computation during training and also helps isolate the comparison of fusion mechanisms from the effect of image encoder fine-tuning. Since the dataset size is limited, the backbone is kept frozen to reduce the risk of overfitting.""",
    )

    text = replace_once(
        text,
        """It should be noted that any learning-based model taking CBC input also receives cbc\\_mask by default. That means, CBC-only, Concat Fusion, and MM-VAE all use the imputed and standardized CBC values, but they can also draw explicitly on each indicator's missingness status. This reduces the information loss that missing-value handling might otherwise introduce. Image-only, by contrast, uses only the image embedding and has no access to CBC missingness information.

As shown in Fig.~\\ref{fig:mmvvae}, the proposed MM-VAE takes CBC features, the missingness indicator vector, and the patient-level image embedding as input. Two separate encoders estimate latent distribution parameters for the two modalities, and a PoE mechanism fuses them into a shared latent space. From the resulting latent variable, the model jointly performs CBC reconstruction, image embedding reconstruction, and anemia classification. Compared with direct feature concatenation, this design enables the model to capture cross-modal consistency and complementarity within a unified representation space, while also supporting inference under missing modalities.""",
        """It should be noted that any learning-based model taking CBC input also receives cbc\\_mask by default. That means, CBC-only, Concat Fusion, and MM-VAE all use the imputed and standardized CBC values, but they can also draw explicitly on each indicator's missingness status. This reduces the information loss that missing-value handling might otherwise introduce. Image-only, by contrast, uses only the image embedding and has no access to CBC missingness information.

As shown in Fig.~\\ref{fig:mmvvae}, the proposed MM-VAE takes the standardized CBC vector, the corresponding cbc\\_mask vector, and the patient-level image embedding as input. In the implementation, when the missingness mask is enabled, the CBC input is formed by concatenating the standardized CBC features and cbc\\_mask along the feature dimension. The CBC encoder is a Gaussian encoder with two fully connected hidden layers of 64 units, each followed by ReLU activation. It outputs the mean and log-variance of the CBC-specific latent posterior. The image encoder receives the 512-dimensional patient-level image embedding and uses two fully connected hidden layers with 256 and 128 units, respectively, followed by ReLU activation, to estimate the mean and log-variance of the image-specific latent posterior.

The modality-specific posterior distributions are fused by a Product-of-Experts mechanism. In addition to the available modality experts, a standard Gaussian prior expert is included in the fusion process. The fused latent distribution is obtained by precision-weighted combination of the posterior means and variances. A latent variable is then sampled from this fused distribution through the reparameterization trick during training, while the posterior mean is used during evaluation. The latent dimension is set to 16 in all MM-VAE experiments.

From the shared latent variable, the model jointly performs CBC reconstruction, image embedding reconstruction, and anemia classification. The CBC decoder consists of a linear layer from the latent space to 64 hidden units followed by ReLU activation and a final linear layer that reconstructs the CBC vector. The image decoder maps the latent variable through a 128-unit hidden layer and reconstructs the 512-dimensional image embedding. The classifier contains a 64-unit hidden layer, ReLU activation, dropout with a rate of 0.2, and a final linear output for binary anemia prediction. Compared with direct feature concatenation, this design allows CBC and image information to be integrated at the latent distribution level while still supporting inference when one modality is unavailable.""",
    )

    text = replace_once(
        text,
        """For the training strategy, the objective function of MM-VAE consists of CBC reconstruction loss, image reconstruction loss, KL divergence loss, and classification loss. Its overall form is given as:

\\begin{equation}
\\mathcal{L}=\\mathcal{L}_{cbc}+\\alpha\\mathcal{L}_{img}+\\beta\\mathcal{L}_{kl}+\\lambda\\mathcal{L}_{cls}
\\label{eq:loss}
\\end{equation}

Here, the image reconstruction loss, KL divergence loss, and classification loss respectively constrain image embedding reconstruction, regularization of the shared latent space, and anemia binary classification performance. Earlier versions of the framework assigned equal weight to all CBC dimensions during reconstruction. The current implementation instead applies a mask-based reconstruction loss for the CBC branch, computed using cbc\\_mask. This means reconstruction error is calculated only over the CBC indicators that were actually observed. This prevents the model from being overly driven by the imputed values themselves during training. All learning models are trained with the Adam optimizer, with early stopping based on validation set performance. To stabilize the learning of the latent variable, MM-VAE applies a KL warm-up strategy in the early stages of training, gradually increasing the weight of the KL term.""",
        """For the training strategy, the objective function of MM-VAE consists of CBC reconstruction loss, image embedding reconstruction loss, KL divergence loss, and classification loss. Its overall form is given as:

\\begin{equation}
\\mathcal{L}=\\mathcal{L}_{cbc}+\\alpha\\mathcal{L}_{img}+\\beta\\mathcal{L}_{kl}+\\lambda\\mathcal{L}_{cls}
\\label{eq:loss}
\\end{equation}

The CBC reconstruction loss is calculated as a mask-based mean squared error using cbc\\_mask, so that reconstruction error is only computed over CBC indicators that were actually observed before imputation. This prevents the model from being overly driven by the imputed values themselves during training. The image reconstruction loss is computed as the mean squared error between the reconstructed image embedding and the cached 512-dimensional patient-level image embedding. The KL divergence regularizes the fused latent posterior toward the standard Gaussian prior, and the classification loss is computed by binary cross-entropy with logits.

In the reported MM-VAE embedding experiments, the model is trained with the Adam optimizer, a learning rate of 0.001, weight decay of 1e-5, batch size of 32, and a maximum of 100 epochs. Early stopping is applied with a patience of 15 epochs according to validation performance. To stabilize latent variable learning, a KL warm-up strategy is used in the first 20 epochs, gradually increasing the KL weight from 0 to its target value. In the final experimental setting, the image reconstruction loss weight, KL loss weight, and classification loss weight are set to 0.1, 0.1, and 10.0, respectively. The final model is selected according to validation ROC-AUC when available, otherwise validation F1-score is used, and the selected checkpoint is evaluated on the independent test set.""",
    )

    text = replace_once(
        text,
        """While there is still formal significance testing to come between MM-VAE and Concat Fusion, the multi-seed results suggest that MM-VAE consistently outperforms Concat Fusion. The benefit of multimodal learning over unimodal screening is clearly established, as is also the additional stable benefit from the shared latent model over simple concatenation.""",
        """While there is still formal significance testing to come between MM-VAE and Concat Fusion, the multi-seed results suggest that MM-VAE consistently outperforms Concat Fusion. The benefit of multimodal learning over unimodal screening is clearly established, as is also the additional stable benefit from the shared latent model over simple concatenation.

This result also helps clarify the practical advantage of the proposed PoE-based fusion strategy. The improvement of Concat Fusion over unimodal models indicates that the two modalities are indeed complementary, but the further gain of MM-VAE suggests that simple feature-level concatenation is not sufficient to fully exploit this complementarity. Instead, integrating CBC and image information through a shared probabilistic latent space appears to provide a more stable and effective fusion mechanism for patient-level anemia screening. Given the relatively small dataset size and the need to handle incomplete inputs, this also supports the suitability of PoE over heavier cross-modal interaction architectures in the present task.""",
    )

    text = replace_once(
        text,
        """Experimental results show that multimodal modeling performs better than unimodal baselines. MM-VAE outperformed Concat Fusion across all six evaluation metrics and exhibited smaller variance across random seeds. The HGB ablation experiment further showed that MM-VAE suffered only limited performance degradation after HGB removal, while the CBC-only baseline deteriorated substantially. It suggests that the model draws on complementary information from the image modality rather than depending primarily on a single dominant diagnostic feature. The missing-modality experiment also showed that the PoE mechanism can maintain reasonable predictive ability when the image modality is absent. Though performance declined more sharply when CBC was removed, indicating that the image modality, in its current form, does not yet carry sufficient standalone discriminative signal to support independent diagnostic decisions --- a limitation attributable to the frozen mean-pooled image representation rather than the fusion mechanism itself.""",
        """Experimental results show that multimodal modeling performs better than unimodal baselines. MM-VAE outperformed Concat Fusion across all six evaluation metrics and exhibited smaller variance across random seeds. This suggests not only that CBC and image modalities are complementary, but also that a probabilistic latent-space fusion strategy is more effective than direct feature-level fusion for the present task. In particular, the PoE mechanism provides a practical way to integrate quantitative hematological evidence and visual morphological evidence within a unified framework, while retaining the ability to operate under missing-modality conditions. The HGB ablation experiment further showed that MM-VAE suffered only limited performance degradation after HGB removal, while the CBC-only baseline deteriorated substantially. It suggests that the model draws on complementary information from the image modality rather than depending primarily on a single dominant diagnostic feature. The missing-modality experiment also showed that the PoE mechanism can maintain reasonable predictive ability when the image modality is absent. Though performance declined more sharply when CBC was removed, indicating that the image modality, in its current form, does not yet carry sufficient standalone discriminative signal to support independent diagnostic decisions --- a limitation attributable to the frozen mean-pooled image representation rather than the fusion mechanism itself.""",
    )

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(text, encoding="utf-8")
    print(OUT)


if __name__ == "__main__":
    main()
