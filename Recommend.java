// Matric Number: A0172430R
// Name: Tang Yurou
// Recommend.java
import java.io.*;
import java.util.*;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.io.*;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.input.KeyValueTextInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.mapreduce.lib.input.MultipleInputs;


public class Recommend {

    public static class ReadMapper
            extends Mapper<Object, Text, IntWritable, Text>{

        private IntWritable outputKey;
        private Text outputValue = new Text();
        String userId, itemId, rating;

        public void map(Object key, Text value, Context context
        ) throws IOException, InterruptedException {
            String[] tokens = value.toString().split(",");
            userId = tokens[0];
            itemId = tokens[1];
            rating = tokens[2];
            outputKey = new IntWritable(Integer.parseInt(userId));
            outputValue.set(itemId + ":" + rating);
            context.write(outputKey, outputValue);
        }
    }

    public static class GroupingReducer
            extends Reducer<IntWritable,Text,IntWritable,Text> {

        public void reduce(IntWritable key, Iterable<Text> values,
                           Context context
        ) throws IOException, InterruptedException {
            StringBuilder string = new StringBuilder();

            string.append(values.iterator().next());
            while (values.iterator().hasNext()){
                string.append(",");
                string.append(values.iterator().next());
            }
            context.write(key, new Text(string.toString()));
        }
    }

    public static class PairMapper
            extends Mapper<LongWritable, Text, Text, IntWritable>{

        private final static IntWritable one = new IntWritable(1);

        public void map(LongWritable key, Text value, Context context
        ) throws IOException, InterruptedException {
            String[] inputs = value.toString().trim().split("\t");
            String ratings = inputs[1];
            String[] itemsList = ratings.split(",");

            for (String itemOne: itemsList) {
                String itemOneId = itemOne.trim().split(":")[0];
                for (String itemTwo: itemsList) {
                    String itemTwoId = itemTwo.trim().split(":")[0];
                    context.write(new Text(itemOneId + ":" + itemTwoId), one);
                }
            }
        }
    }

    public static class CountReducer
            extends Reducer<Text,IntWritable,Text,IntWritable> {

        private IntWritable count = new IntWritable();

        public void reduce(Text key, Iterable<IntWritable> values,
                           Context context
        ) throws IOException, InterruptedException {
            int sum = 0;
            for (IntWritable val : values) {
                sum += val.get();
            }
            count.set(sum);
            context.write(key, count);
        }
    }

    public static class SplitMapper
            extends Mapper<Text, Text, IntWritable, Text>{

        public void map(Text key, Text value, Context context
        ) throws IOException, InterruptedException {
            String count = value.toString();
            String[] items = key.toString().split(":");
            int itemOneId = Integer.parseInt(items[0]);
            context.write(new IntWritable(itemOneId), new Text(items[1] + ":" + count));
        }
    }

    public static class ScoresMapper
            extends Mapper<Text, Text, IntWritable, Text>{

        private IntWritable outputKey;
        private Text outputValue = new Text();
        String userId, itemId, score;

        public void map(Text key, Text value, Context context
        ) throws IOException, InterruptedException {
            String[] tokens = key.toString().split(",");
            userId = tokens[0];
            itemId = tokens[1];
            score = tokens[2];

            outputKey = new IntWritable(Integer.parseInt(itemId));
            outputValue.set("scores:" + userId + ":" + score);
            context.write(outputKey, outputValue);
        }
    }

    public static class MultiplicationReducer
            extends Reducer<IntWritable, Text, Text, DoubleWritable> {

        public void reduce(IntWritable key, Iterable<Text> values,
                           Context context
        ) throws IOException, InterruptedException {

            Map<String, Double> itemScores = new HashMap<>();
            Map<String, Double> userScores = new HashMap<>();

            for (Text val : values) {
                if (val.toString().contains("scores:")) {
                    String[] userScore = val.toString().split(":");
                    userScores.put(userScore[1], Double.parseDouble(userScore[2]));
                } else {
                    String[] itemScore = val.toString().split(":");
                    itemScores.put(itemScore[0], Double.parseDouble(itemScore[1]));
                }
            }
            for (Map.Entry<String, Double> entry: itemScores.entrySet()) {
                String itemId = entry.getKey();
                double itemScore = entry.getValue();
                for (Map.Entry<String, Double> element: userScores.entrySet()) {
                    String userId = element.getKey();
                    double userScore = element.getValue();
                    context.write(new Text(userId + ":" + itemId), new DoubleWritable(itemScore * userScore));
                }
            }
        }
    }

    public static class SumMapper
            extends Mapper<LongWritable, Text, Text, DoubleWritable>{

        public void map(LongWritable key, Text value, Context context
        ) throws IOException, InterruptedException {
            String[] inputs = value.toString().trim().split("\t");
            String outputKey = inputs[0];
            double outputValue = Double.parseDouble(inputs[1]);
            context.write(new Text(outputKey), new DoubleWritable(outputValue));
        }
    }

    public static class SumReducer
            extends Reducer<Text,DoubleWritable,Text,Text> {

        private Set<String> givenScores = new HashSet<>();

        public void setup(Context context) throws IOException {
            String filepath = context.getConfiguration().get("input");
            FileSystem fs = FileSystem.get(context.getConfiguration());
            BufferedReader br = new BufferedReader(new InputStreamReader(fs.open(new Path(filepath))));
            String line;
            try {
                while ((line = br.readLine()) != null) {
                    String[] tokens = line.split(",");
                    givenScores.add(tokens[0] + "," + tokens[1]);
                }
            } catch (IOException e) {
                e.printStackTrace();
            }
        }

        public void reduce(Text key, Iterable<DoubleWritable> values,
                           Context context
        ) throws IOException, InterruptedException {
            String[] userItem = key.toString().trim().split(":");
            String user = userItem[0];
            String item = userItem[1];
            double sum = 0;
            for (DoubleWritable val : values) {
                sum += val.get();
            }
            if (!givenScores.contains(user + "," + item)) {
                context.write(new Text(user), new Text(item + "," + sum));
            }
        }
    }

    public static void main(String[] args) throws Exception {
        Configuration conf = new Configuration();
        conf.set("input", args[0]);

        Job job = Job.getInstance(conf, "group by user");
        job.setJarByClass(Recommend.class);
        job.setMapperClass(ReadMapper.class);
        job.setReducerClass(GroupingReducer.class);
        job.setOutputKeyClass(IntWritable.class);
        job.setOutputValueClass(Text.class);
        FileInputFormat.addInputPath(job, new Path(args[0]));
        FileOutputFormat.setOutputPath(job, new Path("recommendation/1_group"));
        job.waitForCompletion(true);

        Job job2 = Job.getInstance(conf, "create co-occurrence matrix");
        job2.setJarByClass(Recommend.class);
        job2.setMapperClass(PairMapper.class);
        job2.setReducerClass(CountReducer.class);
        job2.setOutputKeyClass(Text.class);
        job2.setOutputValueClass(IntWritable.class);
        FileInputFormat.addInputPath(job2, new Path("recommendation/1_group"));
        FileOutputFormat.setOutputPath(job2, new Path("recommendation/2_co-occurrence"));
        job2.waitForCompletion(true);

        Job job3 = Job.getInstance(conf, "matrix multiplication");
        job3.setJarByClass(Recommend.class);
        job3.setReducerClass(MultiplicationReducer.class);
        MultipleInputs.addInputPath(job3, new Path("recommendation/2_co-occurrence"),
                KeyValueTextInputFormat.class, SplitMapper.class);
        MultipleInputs.addInputPath(job3, new Path(args[0]),
                KeyValueTextInputFormat.class, ScoresMapper.class);
        job3.setMapOutputKeyClass(IntWritable.class);
        job3.setMapOutputValueClass(Text.class);
        job3.setOutputKeyClass(Text.class);
        job3.setOutputValueClass(DoubleWritable.class);
        FileOutputFormat.setOutputPath(job3, new Path("recommendation/3_multiplication"));
        job3.waitForCompletion(true);

        Job job4 = Job.getInstance(conf, "sum matrix");
        job4.setJarByClass(Recommend.class);
        job4.setMapperClass(SumMapper.class);
        job4.setReducerClass(SumReducer.class);
        job4.setOutputKeyClass(Text.class);
        job4.setOutputValueClass(DoubleWritable.class);
        FileInputFormat.addInputPath(job4, new Path("recommendation/3_multiplication"));
        FileOutputFormat.setOutputPath(job4, new Path(args[1]));
        System.exit(job4.waitForCompletion(true) ? 0 : 1);
    }
}
