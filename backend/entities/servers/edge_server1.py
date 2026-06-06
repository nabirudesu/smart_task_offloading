# Updated EdgeServer Logic - Complete Flow After Cost Calculation
# entities/edge_server_updated.py

# Add these imports to the existing edge_server.py
from offloading_algorithme import OffloaingAlgorithme, PPOModel
import time


class EdgeServer(BaseServer):
    # ... existing code ...

    def calculate_task_cost_outputes(
        self, generated_task_data_list: list[Optional[TaskModelPlatform]]
    ):
        """
        Complete pipeline: calculate costs → get offloading decisions → send to vehicles

        Inputs:
        --- List of TaskModelPlatform objects with calculated cost estimations

        Outputs:
        --- Sends offloading decisions back to vehicles for execution
        """
        if not self.cost_model:
            print("[ERROR] Cost model is not set. Cannot calculate task costs.")
            return None

        print(
            f"[TIME {self.env.now:.2f}] Starting cost calculation and offloading decisions for {len(generated_task_data_list)} tasks"
        )

        # 1. Calculate costs for all task combinations
        for tasks_batch in generated_task_data_list:
            if not tasks_batch:
                continue
            self.cost_model.estimate(tasks_batch)
            if self.cloud_server and self.cloud_server.cost_model:
                self.cloud_server.cost_model.estimate(tasks_batch)

        print(f"[TIME {self.env.now:.2f}] Cost estimation completed for all tasks")

        # 2. Prepare inputs for offloading algorithm
        start_time = time.time()
        rl_inputs = self.prepare_offloading_inputs(generated_task_data_list)
        input_preparation_time = time.time() - start_time

        print(f"[TIME {self.env.now:.2f}] RL inputs prepared in {input_preparation_time:.4f}s")

        # Simulate the time needed for input preparation
        yield self.env.timeout(input_preparation_time)

        # 3. Calculate offloading decisions using RL algorithm
        print(f"[TIME {self.env.now:.2f}] Starting offloading decision calculation...")

        decision_start_time = time.time()
        tasks_with_decisions = yield from self._calculate_offloading_decisions_process(
            generated_task_data_list, rl_inputs
        )
        decision_time = time.time() - decision_start_time

        print(f"[TIME {self.env.now:.2f}] Offloading decisions completed in {decision_time:.4f}s")

        # 4. Send results back to vehicles
        yield from self._send_decisions_to_vehicles(tasks_with_decisions)

        print(f"[TIME {self.env.now:.2f}] All tasks processed and sent back to vehicles")

    def _calculate_offloading_decisions_process(
        self, task_batch: list[Optional[TaskModelPlatform]], rl_inputs: dict
    ):
        """
        SimPy process to calculate offloading decisions using the RL algorithm
        """
        # Initialize the offloading algorithm (PPO model)
        algorithm_start_time = time.time()

        try:
            print(f"[TIME {self.env.now:.2f}] Initializing PPO model for offloading decisions...")
            offloading_algo = PPOModel(rl_inputs)

            # Calculate decisions
            tasks_with_decisions = offloading_algo.calculate_offloading_decisions(task_batch)

            algorithm_execution_time = time.time() - algorithm_start_time
            print(
                f"[TIME {self.env.now:.2f}] PPO algorithm executed in {algorithm_execution_time:.4f}s"
            )

            # Simulate the computational time in the simulation
            yield self.env.timeout(algorithm_execution_time)

            # Validate decisions
            valid_decisions_count = 0
            for task in tasks_with_decisions:
                if task and task.chosen_execution:
                    valid_decisions_count += 1
                    print(
                        f"[TIME {self.env.now:.2f}] Task {task.task_id} → {self._get_execution_level_name(task.chosen_execution.level)} on {task.chosen_execution.platform.name}"
                    )
                elif task:
                    print(f"[WARNING] Task {task.task_id} has no chosen execution!")

            print(
                f"[TIME {self.env.now:.2f}] Valid decisions: {valid_decisions_count}/{len([t for t in tasks_with_decisions if t])}"
            )

            return tasks_with_decisions

        except Exception as e:
            print(f"[ERROR] Offloading decision calculation failed: {e}")
            print(f"[TIME {self.env.now:.2f}] Falling back to default decisions...")

            # Fallback: assign local execution for all tasks
            fallback_algo = OffloaingAlgorithme()
            tasks_with_decisions = fallback_algo._fallback_decisions(task_batch)

            yield self.env.timeout(0.1)  # Small delay for fallback processing
            return tasks_with_decisions

    def _send_decisions_to_vehicles(self, tasks_with_decisions: list[Optional[TaskModelPlatform]]):
        """
        SimPy process to send offloading decisions back to vehicles
        """
        print(f"[TIME {self.env.now:.2f}] Sending offloading decisions to vehicles...")

        # Group tasks by vehicle for efficient communication
        vehicle_task_groups = self._group_tasks_by_vehicle(tasks_with_decisions)

        # Send to each vehicle
        vehicle_processes = []
        for vehicle_id, vehicle_tasks in vehicle_task_groups.items():
            if vehicle_tasks:
                process = self.env.process(
                    self._send_to_specific_vehicle(vehicle_id, vehicle_tasks)
                )
                vehicle_processes.append(process)

        # Wait for all vehicle communications to complete
        if vehicle_processes:
            yield self.env.all_of(vehicle_processes)
            print(f"[TIME {self.env.now:.2f}] All offloading decisions sent to vehicles")
        else:
            print(f"[TIME {self.env.now:.2f}] No valid vehicle communications to send")

    def _group_tasks_by_vehicle(
        self, tasks_with_decisions: list[Optional[TaskModelPlatform]]
    ) -> dict[int, list[TaskModelPlatform]]:
        """
        Group tasks by their source vehicle for efficient communication
        """
        vehicle_groups: dict[int, list[TaskModelPlatform]] = {}

        for task in tasks_with_decisions:
            if task and task.vehicle:
                vehicle_id = task.vehicle.id
                if vehicle_id not in vehicle_groups:
                    vehicle_groups[vehicle_id] = []
                vehicle_groups[vehicle_id].append(task)

        print(
            f"[TIME {self.env.now:.2f}] Grouped {len([t for t in tasks_with_decisions if t])} tasks into {len(vehicle_groups)} vehicle groups"
        )
        return vehicle_groups

    def _send_to_specific_vehicle(self, vehicle_id: int, vehicle_tasks: list[TaskModelPlatform]):
        """
        SimPy process to send offloading decisions to a specific vehicle
        """
        try:
            # Find the vehicle object (this assumes you have access to vehicles)
            target_vehicle = None
            for task in vehicle_tasks:
                if task.vehicle and task.vehicle.id == vehicle_id:
                    target_vehicle = task.vehicle
                    break

            if not target_vehicle:
                print(f"[ERROR] Vehicle {vehicle_id} not found for task communication")
                return

            print(
                f"[TIME {self.env.now:.2f}] Sending {len(vehicle_tasks)} decisions to Vehicle {vehicle_id}"
            )

            # Calculate communication data size (decision metadata)
            decision_data_size = len(vehicle_tasks) * 100  # Assume 100 bytes per decision

            # Simulate network transmission time
            transmission_time = decision_data_size / self.network.edge_to_vehicle_throughput
            transmission_energy = self.power_P_e_v * transmission_time

            print(
                f"[TIME {self.env.now:.2f}] Transmitting {decision_data_size} bytes to Vehicle {vehicle_id} (estimated {transmission_time:.4f}s)"
            )

            # Simulate transmission delay
            yield self.env.timeout(transmission_time)

            # Call vehicle's offloading function
            print(f"[TIME {self.env.now:.2f}] Triggering task execution on Vehicle {vehicle_id}")

            # Start the vehicle's offloading process (non-blocking)
            self.env.process(target_vehicle.offload_task_to_destination(vehicle_tasks))

            print(f"[TIME {self.env.now:.2f}] Successfully sent decisions to Vehicle {vehicle_id}")

        except Exception as e:
            print(f"[ERROR] Failed to send decisions to Vehicle {vehicle_id}: {e}")

    def _get_execution_level_name(self, level) -> str:
        """
        Convert execution level enum to readable string
        """
        if hasattr(level, "name"):
            return level.name.lower()
        return str(level)

    # Enhanced logging for better monitoring
    def _log_batch_processing_summary(
        self, tasks_with_decisions: list[Optional[TaskModelPlatform]]
    ):
        """
        Log summary statistics for the processed batch
        """
        if not tasks_with_decisions:
            return

        valid_tasks = [t for t in tasks_with_decisions if t and t.chosen_execution]

        level_counts = {"Vehicle": 0, "Edge": 0, "Cloud": 0}
        for task in valid_tasks:
            level_name = self._get_execution_level_name(task.chosen_execution.level)
            if level_name in level_counts:
                level_counts[level_name] += 1

        print(f"[TIME {self.env.now:.2f}] BATCH SUMMARY:")
        print(f"  - Total tasks processed: {len(valid_tasks)}")
        print(f"  - Vehicle executions: {level_counts['Vehicle']}")
        print(f"  - Edge executions: {level_counts['Edge']}")
        print(f"  - Cloud executions: {level_counts['Cloud']}")
        print(
            f"  - Tasks with decisions: {len(valid_tasks)}/{len([t for t in tasks_with_decisions if t])}"
        )

    # Updated generate_edge_tasks to include logging
    def generate_edge_tasks(self, tasks_batch: list[Optional[TaskModelPlatform]]):
        """
        Enhanced task generation with complete pipeline execution
        """
        print(f"[TIME {self.env.now:.2f}] Processing batch of {len(tasks_batch)} tasks")

        for task in tasks_batch:
            if not task:
                continue
            _type = task.task.type
            task_id = task.task.id

            # Generate executions for edge server
            for model in self.deployed_models:
                if model.type == _type:
                    for platform in self.processing_platforms_list:
                        task.append_execution(
                            ModelPlatformExecution(self.level, task_id, model, platform)
                        )

            # Generate executions for cloud server
            if self.cloud_server:
                for model in self.cloud_server.deployed_models:
                    if model.type == _type:
                        for platform in self.cloud_server.processing_platforms_list:
                            task.append_execution(
                                ModelPlatformExecution(
                                    self.cloud_server.level, task_id, model, platform
                                )
                            )

            # Calculate future time and position for alpha
            self.calculate_future_time(task, self.network)
            self.calculate_future_position(task)

        # Execute the complete pipeline
        yield from self.calculate_task_cost_outputes(tasks_batch)

        # Log processing summary
        self._log_batch_processing_summary(tasks_batch)

        print(f"[TIME {self.env.now:.2f}] Batch processing completed successfully")

    # Enhanced batch extractor with better error handling
    def batch_extractor_agent(self):
        """
        Enhanced batch extractor with improved error handling and logging
        """
        print(f"Time {self.env.now:.2f}: [AGENT] Batch extractor agent started for {self.name}.")

        while True:
            try:
                # Wait for the time slot
                print(f"\nTime {self.env.now:.2f}: [AGENT] Sleeping for {TIME_SLOT}ms.")
                yield self.env.timeout(TIME_SLOT)

                # Check for tasks
                if not self.task_queue.items:
                    print(f"Time {self.env.now:.2f}: [AGENT] No tasks in queue. Continuing...")
                    continue

                print(
                    f"Time {self.env.now:.2f}: [AGENT] Extracting batch of up to {MAX_BATCH_SIZE} tasks."
                )

                # Extract batch
                batch_process = self.env.process(self._extract_tasks_from_store(MAX_BATCH_SIZE))
                batch = yield batch_process

                if batch:
                    task_ids = [task.task_id for task in batch if task]
                    print(
                        f"Time {self.env.now:.2f}: [AGENT] Extracted {len(batch)} tasks: {task_ids}"
                    )
                    print(
                        f"Time {self.env.now:.2f}: [AGENT] Remaining in queue: {len(self.task_queue.items)}"
                    )

                    # Process the batch with complete pipeline
                    yield from self.generate_edge_tasks(batch)
                else:
                    print(f"Time {self.env.now:.2f}: [AGENT] No tasks extracted this cycle.")

            except Interrupt as i:
                print(f"Time {self.env.now:.2f}: [AGENT] Agent interrupted: {i.cause}")
                break
            except Exception as e:
                print(f"Time {self.env.now:.2f}: [AGENT] Unexpected error: {e}")
                # Continue processing despite errors
                continue
