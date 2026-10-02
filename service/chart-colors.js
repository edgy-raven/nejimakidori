// Custom scalar charts otherwise assign one color to every metric in a run.
const metricColors = {
  total_loss: '#0072b2',
  policy_loss: '#d55e00',
  discard_match: '#0072b2',
  discard_top3: '#d55e00',
  'training_parameters/offline_payoff_weight': '#0072b2',
  'training_parameters/offline_policy_mix': '#d55e00',
  payoff_critic_loss: '#0072b2',
  payoff_baseline_loss: '#d55e00',
  expert_attack_imitation_loss: '#0072b2',
  expert_defense_imitation_loss: '#d55e00',
  round_placement_loss: '#0072b2',
  final_placement_loss: '#d55e00',
  payment_cross_entropy: '#0072b2',
  payment_occurrence_loss: '#d55e00',
  opponent_shanten_loss: '#0072b2',
  opponent_hand_count_loss: '#d55e00',
  opponent_ukeire_loss: '#0072b2',
  tenpai_ukeire_loss: '#009e73',
  other_ukeire_loss: '#d55e00',
  completion_yaku_loss: '#0072b2',
  tenpai_wait_recall_at_5: '#0072b2',
  'training_parameters/learning_rate': '#0072b2',
  'training_parameters/replica_gradient_norm': '#009e73',
  'training_parameters/replica_actor_gradient_norm': '#0072b2',
  'training_parameters/replica_value_gradient_norm': '#d55e00',
  'training_parameters/examples_per_second': '#0072b2',
};

customElements.whenDefined('tf-custom-scalar-multi-line-chart-card').then(() => {
  const card = customElements.get('tf-custom-scalar-multi-line-chart-card');
  const updateChart = card.prototype._updateChart;
  const metricScale = {scale: name => metricColors[name.split(' (')[0]] || '#64748b'};
  card.prototype._updateChart = function () {
    this._colorScale = metricScale;
    return updateChart.call(this);
  };
});
