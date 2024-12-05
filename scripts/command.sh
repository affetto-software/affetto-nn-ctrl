# Plot score for specific trained model
## Run model training and score calculation.
merge=; for i in {00..15}; do label="left_elbow"; cont=step; sync=sync; scale=fast; a=preview-ref; ap=step${i}; s=none; r=linear; rp=default; dataset="/home/atsuta/Dropbox/work/data/affetto_nn_ctrl/dataset/${label}/20241203T142723/${cont}/${sync}/${scale}"; rye run python apps/train_model.py -v "${dataset}" -j 5 --test-size 0.25 --seed 42 -m "/home/atsuta/Dropbox/work/data/affetto_nn_ctrl/config/model.toml" -a "${a}.${ap}" -s "${s}" -r "${r}.${rp}" --label "${label}" --sublabel "${a}.${ap}/${r}.${rp}/${s}/${cont}_${sync}_${scale}" ${merge:+--specify-date} ${merge:+latest} && rye run python apps/calculate_score.py -v "/home/atsuta/Dropbox/work/data/affetto_nn_ctrl/trained_model/${label}/latest/${a}.${ap}/${r}.${rp}/${s}/${cont}_${sync}_${scale}/trained_model.joblib" -d "${dataset}" --test-size 0.25 --seed 42 -e png pdf --no-show-screen; merge=yes; done
## Make plot
cont=step; sync=sync; scale=fast; a=preview-ref; ap=step00; s=none; r=linear; rp=default; rye run python apps/drawer/compare_scores_across_steps.py "/home/atsuta/Dropbox/work/data/affetto_nn_ctrl/trained_model/left_elbow/latest" -a "${a}" -s "${s}" -r "${r}.${rp}" -d "${cont}_${sync}_${scale}" -e png pdf

# Compare scores between scalers
## Train models and calculate scores
merge=; for s in none std minmax maxabs robust; do for i in {00..15}; do label="left_elbow"; cont=step; sync=sync; scale=fast; a=preview-ref; ap=step${i}; r=linear; rp=default; dataset="/home/atsuta/Dropbox/work/data/affetto_nn_ctrl/dataset/${label}/20241203T142723/${cont}/${sync}/${scale}"; echo "-----"; rye run python apps/train_model.py -v "${dataset}" -j 5 --test-size 0.25 --seed 42 -m "/home/atsuta/Dropbox/work/data/affetto_nn_ctrl/config/model.toml" -a "${a}.${ap}" -s "${s}" -r "${r}.${rp}" --label "${label}" --sublabel "${a}.${ap}/${r}.${rp}/${s}/${cont}_${sync}_${scale}" ${merge:+--specify-date} ${merge:+latest} && rye run python apps/calculate_score.py -v "/home/atsuta/Dropbox/work/data/affetto_nn_ctrl/trained_model/${label}/latest/${a}.${ap}/${r}.${rp}/${s}/${cont}_${sync}_${scale}/trained_model.joblib" -d "${dataset}" --test-size 0.25 --seed 42 -e png pdf --no-show-screen; merge=yes; done; done

# Variables
adapters=(preview-ref delay-states delay-states-all)
scalers=(none std minmax maxabs robust)
regressors=(
  linear.default
  ridge.default
  ridge.alpha05
  ridge.alpha01
  mlp.default
  mlp.default-iter500
  mlp.default-iter800
  mlp.layer200
  mlp.layer200-iter500
  mlp.layer200-iter800
  mlp.layer100-100
  mlp.layer100-100-iter500
  mlp.layer100-100-iter800
)

# Train models to find optimized steps for all patterns
merge=; for _a in "${adapters[@]}"; do for r in "${regressors[@]}"; do for s in "${scalers[@]}"; do for i in {00..15}; do label="left_elbow"; cont=step; sync=sync; scale=fast; a="${_a}.step${i}"; dataset="/home/atsuta/Dropbox/work/data/affetto_nn_ctrl/dataset/${label}/20241203T142723/${cont}/${sync}/${scale}"; dataset_tag="${cont}_${sync}_${scale}"; echo "-----"; rye run python apps/train_model.py -v "${dataset}" -j 5 --test-size 0.25 --seed 42 -m "/home/atsuta/Dropbox/work/data/affetto_nn_ctrl/config/model.toml" -a "${a}" -s "${s}" -r "${r}" --label "${label}" --sublabel "${a}/${r}/${s}/${dataset_tag}" ${merge:+--specify-date} ${merge:+latest} && rye run python apps/calculate_score.py -v "/home/atsuta/Dropbox/work/data/affetto_nn_ctrl/trained_model/${label}/latest/${a}/${r}/${s}/${dataset_tag}/trained_model.joblib" -d "${dataset}" --test-size 0.25 --seed 42 -e png pdf --no-show-screen || echo "${label} ${a}/${r}/${s}/${dataset_tag}$" >>failure.log; merge=yes; done; done; done; done


## Make plot for comparison across adapters
cont=step; sync=sync; scale=fast; s=none; r=linear.default; dataset_tag="${cont}_${sync}_${scale}"; rye run python apps/drawer/compare_scores_across_steps.py "/home/atsuta/Dropbox/work/data/affetto_nn_ctrl/trained_model/left_elbow/20241205T085446" -a "${adapters[@]}" -s "${s}" -r "${r}" -d "${dataset_tag}"
## Make plot for comparison across scalers
cont=step; sync=sync; scale=fast; a=preview-ref; r=linear.default; dataset_tag="${cont}_${sync}_${scale}"; rye run python apps/drawer/compare_scores_across_steps.py "/home/atsuta/Dropbox/work/data/affetto_nn_ctrl/trained_model/left_elbow/20241205T085446" -a "${a}" -s "${scalers[@]}" -r "${r}" -d "${dataset_tag}"
## Make plot for comparison across regressors
cont=step; sync=sync; scale=fast; a=preview-ref; s=none; dataset_tag="${cont}_${sync}_${scale}"; rye run python apps/drawer/compare_scores_across_steps.py "/home/atsuta/Dropbox/work/data/affetto_nn_ctrl/trained_model/left_elbow/20241205T085446" -a "${a}" -s "${s}" -r "${regressors[@]}" -d "${dataset_tag}"


# Local Variables:
# jinx-local-words: "atsuta ctrl dataset joblib nn regressor regressors scalers"
# End:
