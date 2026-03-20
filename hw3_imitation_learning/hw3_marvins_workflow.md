# Workflow / Commands used for HW3

## Exercise 1
1. Record expert demonstrations
    ```bash
    python scripts/record_teleop_demos.py
    ```
2. Compute actions
    ```bash
    python scripts/compute_actions.py --action-space ee
    ```
3. Train the model
    ```bash
    python scripts/train.py \
    --exercise 1 \
    --zarr datasets/processed/single_cube/processed_ee_xyz.zarr \
    --state-keys state_ee_xyz state_gripper "state_cube[:3]" state_obstacle \
    --action-keys action_ee_xyz action_gripper \
    --policy obstacle
    ```
4. Rename the checkpoint
    ```bash
    mv checkpoints/single_cube/best_model_ee_xyz_obstacle.pt checkpoints/single_cube/ex1.pt
    ```
5. (Optional) Evaluate the model
    ```bash
    python scripts/eval.py --checkpoint checkpoints/single_cube/ex1.pt
    ```
    Or headless:
    ```bash
    python scripts/eval.py --checkpoint checkpoints/single_cube/ex1.pt --headless --num-episodes 100
    ```
6. Generate the submission file
    ```bash
    python student_eval/run_eval.py --exercise 1 --checkpoint checkpoints/single_cube/ex1.pt
    ```

## Exercise 2
1. Assert that exercise 1 is done and `checkpoints/single_cube/ex1.pt` exists.
2. Record dagger expert demonstrations for `adversarial` setting (out of distribution)
    ```bash
    python scripts/dagger_eval.py \
    --checkpoint checkpoints/single_cube/ex1.pt \
    --num-episodes 10
    ```
3. Compute actions for dagger dataset
    ```bash
    python scripts/compute_actions.py --action-space ee
    ```
4. Retrain the model with the newly merged dataset
    ```bash
    python scripts/train.py \
    --exercise 2 \
    --zarr datasets/processed/single_cube/processed_ee_xyz.zarr \
    --state-keys state_ee_xyz state_gripper "state_cube[:3]" state_obstacle \
    --action-keys action_ee_xyz action_gripper \
    --policy obstacle
    ```
5. Take a note of how the resulting checkpoint is called (e.g. `checkpoints/single_cube/best_model_ee_xyz_obstacle_dagger10ep.pt`)
6. Evaluate the new model in adversarial setting:
    ```bash
    python scripts/eval.py \
    --checkpoint checkpoints/single_cube/best_model_ee_xyz_obstacle_dagger10ep.pt \
    --adversarial \
    --headless \
    --num-episodes 100
    ```
7. (Optional) record more dagger expert demonstrations **with the newest model as the policy**, merge datasets and retrain
    ```
    python scripts/dagger_eval.py \
    --checkpoint checkpoints/single_cube/best_model_ee_xyz_obstacle_dagger10ep.pt \
    --num-episodes 10
    ```
    ```bash
    python scripts/compute_actions.py --action-space ee
    ```
    ```bash
    python scripts/train.py \
    --exercise 2 \
    --zarr datasets/processed/single_cube/processed_ee_xyz.zarr \
    --state-keys state_ee_xyz state_gripper "state_cube[:3]" state_obstacle \
    --action-keys action_ee_xyz action_gripper \
    --policy obstacle
    ```
    *Note: Training is always done on the whole merged dataset but when recording dagger demonstrations, it matters which model is used as the policy*
8. Rename the final checkpoint to ex2.pt
9. Generate the submission file
    ```bash
    python student_eval/run_eval.py --exercise 2 --checkpoint checkpoints/single_cube/ex2.pt
    ```

## Exercise 3
1. Record expert demonstrations
    ```bash
    python scripts/record_teleop_demos.py --multicube
    ```
2. Compute actions
    ```bash
    python scripts/compute_actions.py --action-space ee --datasets-dir ./datasets/raw/multi_cube
    ```
3. Train the model
    ```bash
    python scripts/train.py \
    --exercise 3 \
    --zarr datasets/processed/multi_cube/processed_ee_xyz.zarr \
    --state-keys state_ee_xyz state_gripper "original_pos_cube_red[:3]" "original_pos_cube_green[:3]" "original_pos_cube_blue[:3]" state_goal goal_pos \
    --action-keys action_ee_xyz action_gripper \
    --policy multitask
    ```
4. Rename the checkpoint
    ```bash
    mv checkpoints/multi_cube/best_model_ee_xyz_multitask.pt checkpoints/multi_cube/ex3.pt
    ```
5. (Optional) Evaluate the model
    ```bash
    python scripts/eval.py \
    --checkpoint checkpoints/multi_cube/ex3.pt \
    --multicube \
    --headless \
    --num-episodes 100
    ```
6. Record dagger expert demonstrations with the trained model as the policy:
    *Note: I had to significantly modify dagger_eval.py to make it work in the multicube environment*
    ```bash
    python scripts/dagger_eval.py \
    --checkpoint checkpoints/multi_cube/ex3.pt \
    --multicube \
    --num-episodes 10
    ```
7. Compute actions to merge with dagger:
    ```bash
    python scripts/compute_actions.py --action-space ee --datasets-dir ./datasets/raw/multi_cube
    ```
8. Retrain with the merged dataset
    ```
    python scripts/train.py \
    --exercise 3 \
    --zarr datasets/processed/multi_cube/processed_ee_xyz.zarr \
    --state-keys state_ee_xyz state_gripper "original_pos_cube_red[:3]" "original_pos_cube_green[:3]" "original_pos_cube_blue[:3]" state_goal goal_pos \
    --action-keys action_ee_xyz action_gripper \
    --policy multitask
    ```
9. (Optional) Evaluate again
    ```bash
    python scripts/eval.py \
    --checkpoint checkpoints/multi_cube/best_model_ee_xyz_multitask_dagger10ep.pt \
    --multicube \
    --headless \
    --num-episodes 100
    ```
10. (Optional) Record more dagger demonstrations, retrain, evaluate etc.
11. Rename the final checkpoint to ex3.pt
    ```bash
    mv checkpoints/multi_cube/best_model_ee_xyz_multitask_dagger20ep.pt checkpoints/multi_cube/ex3.pt
    ```
12. Generate the submission file
    ```bash
    python student_eval/run_eval.py --exercise 3 --checkpoint checkpoints/multi_cube/ex3.pt
    ```
