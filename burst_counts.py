import pandas as pd
import numpy as np
import seaborn as sns 
import matplotlib.pyplot as plt 
from scipy import stats 
from scipy.stats import pearsonr
import os

def burst_counts_analysis(filepath): 
    df = pd.read_csv(filepath)

    def category(score): 
        if 0 <= score <= 30: 
            return 'Low'
        elif 30 < score < 70: 
            return 'Mid'
        elif 70 <= score <= 100: 
            return 'High'
    
    df['Stress Group'] = df['Stress Score'].apply(category)


    home = os.path.expanduser('~')
    plot_folder = os.path.join(home, 'Downloads', 'blink_project', 'plots')

    #scatter plot with regression line
    plt.figure()
    sns.regplot(x='burst counts', y='Stress Score', data=df)
    plt.title('')
    plt.xlabel('burst counts')
    plt.ylabel('stress score')
    plt.savefig(os.path.join(plot_folder, f'{os.path.basename(filepath)}_regscatter_burstCounts.png'))
    plt.close()

    #boxplot
    plt.figure()
    sns.boxplot(x='Stress Group', y='burst counts', data=df)
    plt.title('distribution of burst counts across each stress group')
    plt.xlabel('Stress Group')
    plt.ylabel('burst counts')
    plt.savefig(os.path.join(plot_folder, f'{os.path.basename(filepath)}_boxplot_burstCounts.png'))
    plt.close()

    #平均値、標準偏差、個数
    groupData = df.groupby('Stress Group')['burst counts'].agg(['mean', 'std', 'count'])

    mean_High = groupData.loc['High', 'mean'] if 'High' in groupData.index else np.nan
    mean_Mid = groupData.loc['Mid', 'mean'] if 'Mid' in groupData.index else np.nan
    mean_Low = groupData.loc['Low', 'mean'] if 'Low' in groupData.index else np.nan

    std_High = groupData.loc['High', 'std'] if 'High' in groupData.index else np.nan
    std_Mid = groupData.loc['Mid', 'std'] if 'Mid' in groupData.index else np.nan
    std_Low = groupData.loc['Low', 'std'] if 'Low' in groupData.index else np.nan

    count_High = groupData.loc['High', 'count'] if 'High' in groupData.index else 0
    count_Mid = groupData.loc['Mid', 'count'] if 'Mid' in groupData.index else np.nan
    count_Low = groupData.loc['Low', 'count'] if 'Low' in groupData.index else np.nan

    #平均値のみ
    high_stress_group = df[df['Stress Group'] == 'High']['burst counts']
    low_stress_group = df[df['Stress Group'] == 'Low']['burst counts']

    high_stress_mean = np.mean(high_stress_group)
    low_stress_mean = np.mean(low_stress_group)
    
    #t stats and p value 
    t_stats, p_value_tStats = stats.ttest_ind(high_stress_group, low_stress_group, equal_var=False)
    significance_tStats = 'Significant' if p_value_tStats < 0.05 else 'Not Significant'

    #pearson correlation coefficient
    r_value, p_value_pearson = pearsonr(df['burst counts'], df['Stress Score'])
    significance_pearson = 'Significant' if p_value_pearson < 0.05 else 'Not Significant'

    
    result = {
         'Mean for High Stress Group': mean_High, 
         'Mean for Mid Stress Group': mean_Mid, 
         'Mean for Low Stress Group': mean_Low,
         'Std for High Stress Group': std_High, 
         'Std for Mid Stress Group': std_Mid, 
         'Std for Low Stress Group': std_Low, 
         'Count for High Stress Group': count_High, 
         'Count for Mid Stress Group': count_Mid, 
         'Count for Low Stress Group': count_Low, 
         'T statistics': t_stats, 
         'P Value (t-test)': p_value_tStats, 
         'Conclusion for t stats': significance_tStats, 
         'Pearson Correlation Coefficient': r_value, 
         'P Value (Pearson)': p_value_pearson, 
         'Conclusion for pearson': significance_pearson
    }

    summary_folder = os.path.join(home, 'Downloads', 'blink_project', 'summary')
    summary_file = os.path.join(summary_folder, 'burst_counts_result.csv')
    pd.DataFrame([result]).to_csv(summary_file, mode='w', index=False)
    print(f'The csv file has been successfully produced: {os.path.basename(filepath)}')

if __name__ == "__main__": 
    filepath = 'summary.csv'
    burst_counts_analysis(filepath)
