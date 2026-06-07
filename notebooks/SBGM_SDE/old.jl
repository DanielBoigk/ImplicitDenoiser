
function train_epochs!(train_state, learning_rates::Vector{Float32}, print_intermediate,args...)
    num_epochs = length(learning_rates)
    all_losses = Float32[]
    epoch_avg_losses = Float32[]
    
    total_start_time = now()
    println("🏁 Starting training run for $num_epochs epochs...")
    
    for epoch in 1:num_epochs
        epoch_start_time = now()
        dataloader = create_dataloader(args...)
        # 1. Dynamically decay/adjust the learning rate for this epoch
        current_lr = learning_rates[epoch]
        # This adjusts the internal η value inside your NAdam optimizer state
        train_state = adjust!(train_state, current_lr)
        
        println("\n" * "="^60)
        println("🚀 Epoch $epoch / $num_epochs Started")
        println("📅 Time: $(Dates.format(epoch_start_time, "yyyy-mm-dd HH:MM:SS"))")
        println("💧 Current Learning Rate: $current_lr")
        println("="^60)
        
        epoch_losses = Float32[]
        num_batches = length(dataloader)
        
        # 2. Batch loop
        for (i, data) in enumerate(dataloader)
            _, loss, _, train_state = train!(data, train_state)
            
            current_loss = Float32(loss)
            push!(epoch_losses, current_loss)
            push!(all_losses, current_loss)
            
            # Print progress every 10 batches (and final batch)
            if print_intermediate
                if i % 10 == 0 || i == num_batches
                    println("  ⏳ Batch $i / $num_batches | Loss: $(round(current_loss, sigdigits=5))")
                end
            end
        end
        
        epoch_end_time = now()
        duration = CanonicalDimension(epoch_end_time - epoch_start_time)
        avg_loss = mean(epoch_losses)
        push!(epoch_avg_losses, avg_loss)
        
        println("-"^60)
        println("✅ Epoch $epoch Finished at: $(Dates.format(epoch_end_time, "yyyy-mm-dd HH:MM:SS"))")
        println("⏱️  Duration: $duration")
        println("📊 Average Epoch Loss: $(round(avg_loss, sigdigits=5))")
        println("="^60)
    end
    
    total_end_time = now()
    total_duration = total_end_time - total_start_time
    
    println("\n🎉 Training Complete!")
    println("⏱️  Total Execution Time: $total_duration")
    
    # 3. Plot the final training progress directly in your shell
    println("\n📈 Final Loss Trend Across All Batches:")
    plt = lineplot(
        all_losses, 
        title = "Global Training Loss", 
        xlabel = "Global Batch Step", 
        ylabel = "MSE", 
        border = :dotted,
        width = 65,
        height = 12
    )
    display(plt)
    
    return train_state, all_losses, epoch_avg_losses
end

# Decays smoothly over 20 epochs down to roughly 10% of the initial value
num_epochs = 50
base_lr = 1f-3
decay_rate = 0.88f0
lrs = Float32[base_lr * (decay_rate ^ (epoch - 1)) for epoch in 1:num_epochs]



# Run it!
train_state, batch_losses, epoch_losses = train_epochs!(
    train_state, 
    lrs, false, 
    imgs, T, forward, emb_dim, dev, batch_size
)


ps = train_state.parameters
st = train_state.states
@save "ps_latestvn.jld2" ps        
@save "st_latestvn.jld2" st