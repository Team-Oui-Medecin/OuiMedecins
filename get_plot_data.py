from inspect_ai.analysis import evals_df, prepare, log_viewer, model_info
from inspect_viz.view import scores_by_task
from inspect_viz.plot import write_html, write_png

# directory with your .log/.eval files
log_dir = "docs/logs"

# read a DataFrame summarizing each eval
df = evals_df(log_dir)

# print(df.head())
# print(df.shape)
# print(df.columns)
# # get first
# print(df.iloc[0])
# exit()

# (optional) prepare extra columns for plotting
# df = prepare(df, [
#     model_info(),
#     log_viewer("eval", {"logs": "file://" + log_dir})
# ])

# write out a parquet file
df.to_parquet("evals.parquet")

# create and save the plot

from inspect_viz import Data
from inspect_viz.view import scores_by_task
from inspect_viz.plot import write_html

evals = Data.from_file("evals.parquet")
plot = scores_by_task(evals, ci=False)
# write_html("plot.html", plot)
write_png("plot.png", plot)