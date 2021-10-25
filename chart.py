import data
import optimizer
import pandas as pd
import matplotlib.pyplot as plt

data_org = data.RelationshipMap()

sa_org = optimizer.SA(data_org.table_count, data_org.guest_list, data_org.relationships_mat)
pos_current, cost_old, audit_trail = sa_org.anneal()

# visualization
audit_df = pd.DataFrame(audit_trail, columns=['cost_new', 'cost_old', 'temp', 'p_accept'])
audit_df[['cost_old']].plot()
audit_df[['temp']].plot()
audit_df[['p_accept']].plot()
plt.show()
