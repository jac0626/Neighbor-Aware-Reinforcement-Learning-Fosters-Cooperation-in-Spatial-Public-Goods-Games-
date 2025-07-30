import numpy as np
import matplotlib.pyplot as plt

# --- 1. Define theoretical calculation functions ---

def calculate_payoff_advantage(r, w_p):
    """
    Calculate the weighted normalized payoff advantage of defectors.
    Formula: w_p * (P'_D(r) - P'_C(r)) = w_p * (5 - r) / (3r + 5)
    
    Args:
        r (np.array or float): Synergy factor
        w_p (float): Payoff weight
        
    Returns:
        np.array or float: Payoff advantage value
    """
    # Avoid negative r or division by zero
    r = np.asarray(r)
    # Ensure r > -5/3 to avoid division by zero or negative denominator
    if np.any(r <= -5/3):
        raise ValueError("Synergy factor 'r' must be greater than -5/3.")
        
    advantage = w_p * (5 - r) / (3 * r + 5)
    return advantage

def calculate_reputation_subsidy(w_p):
    """
    Calculate the reputation subsidy for cooperators.
    Formula: 1 - w_p
    
    Args:
        w_p (float): Payoff weight
        
    Returns:
        float: Reputation subsidy value
    """
    return 1 - w_p

# --- 2. Plotting settings ---

# Set global font and style for a more paper-friendly look
plt.style.use('seaborn-v0_8-whitegrid') # Use a clean grid style
plt.rcParams.update({
    'font.size': 12,
    'axes.labelsize': 14,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 12,
    'figure.titlesize': 16,
    'font.family': 'serif', # Use a serif font for a more formal look
    'mathtext.fontset': 'dejavuserif'
})

# Create a figure and two subplots (ax1, ax2)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
fig.suptitle('Theoretical Analysis of Reward Competition', fontsize=16, y=1.02)

# Define the range of r
r_values = np.linspace(1.0, 5.0, 400) # Generate 400 points from 1 to 5 for a smooth curve

# --- 3. Plot the first subplot (a) Critical Case: w_p = 0.8 ---

w_p_critical = 0.83
payoff_advantage_critical = calculate_payoff_advantage(r_values, w_p_critical)
reputation_subsidy_critical = calculate_reputation_subsidy(w_p_critical)

# Plot the curves
ax1.plot(r_values, payoff_advantage_critical, label="Defector's Payoff Advantage", color='orangered', linewidth=2.5)
ax1.axhline(y=reputation_subsidy_critical, label="Cooperator's Reputation Subsidy", color='dodgerblue', linestyle='--', linewidth=2.5)

# Find the intersection point
intersection_idx = np.argwhere(np.diff(np.sign(payoff_advantage_critical - reputation_subsidy_critical))).flatten()
if intersection_idx.size > 0:
    r_intersect = r_values[intersection_idx[0]]
    y_intersect = reputation_subsidy_critical
    ax1.plot(r_intersect, y_intersect, 'ko', markersize=8, label=f'Intersection ($r \approx {r_intersect:.2f}$)')
    # Add a vertical line to indicate the intersection point
    ax1.vlines(r_intersect, 0, y_intersect, colors='gray', linestyles='dotted', linewidth=1.5)

# Fill the area to enhance visualization
ax1.fill_between(r_values, payoff_advantage_critical, reputation_subsidy_critical, 
                 where=(reputation_subsidy_critical > payoff_advantage_critical), 
                 color='dodgerblue', alpha=0.2, label='Cooperation Favored')
ax1.fill_between(r_values, payoff_advantage_critical, reputation_subsidy_critical, 
                 where=(payoff_advantage_critical > reputation_subsidy_critical), 
                 color='orangered', alpha=0.2, label='Defection Favored')


# Set subplot title and labels
ax1.set_title(f'(a) Critical Case: $w_P = {w_p_critical}$', fontsize=14)
ax1.set_xlabel('Synergy Factor ($r$)')
ax1.set_ylabel('Reward Component Value')
ax1.legend(loc='upper right')
ax1.set_xlim(1, 5)
ax1.set_ylim(0, 0.4)


# --- 4. Plot the second subplot (b) Normal Case: w_p = 0.7 ---

w_p_normal = 0.7
payoff_advantage_normal = calculate_payoff_advantage(r_values, w_p_normal)
reputation_subsidy_normal = calculate_reputation_subsidy(w_p_normal)

# Plot the curves
ax2.plot(r_values, payoff_advantage_normal, label="Defector's Payoff Advantage", color='orangered', linewidth=2.5)
ax2.axhline(y=reputation_subsidy_normal, label="Cooperator's Reputation Subsidy", color='dodgerblue', linestyle='--', linewidth=2.5)

# Fill the area
ax2.fill_between(r_values, payoff_advantage_normal, reputation_subsidy_normal, 
                 where=(reputation_subsidy_normal > payoff_advantage_normal), 
                 color='dodgerblue', alpha=0.2, label='Cooperation Favored')


# Set subplot title and labels
ax2.set_title(f'(b) Normal Case: $w_P = {w_p_normal}$', fontsize=14)
ax2.set_xlabel('Synergy Factor ($r$)')
# ax2.set_ylabel('Reward Component Value') # Y-axis is shared, no need to set it again
ax2.legend(loc='upper right')
ax2.set_xlim(1, 5)

# --- 5. Adjust layout and save the image ---

plt.tight_layout(rect=[0, 0, 1, 0.96]) # Adjust layout to prevent title overlap
plt.savefig('reward_competition_analysis.png', dpi=300, bbox_inches='tight')
plt.show()