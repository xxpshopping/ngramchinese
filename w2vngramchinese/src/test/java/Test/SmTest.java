package Test;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.OutputStreamWriter;
import java.io.UnsupportedEncodingException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;

import com.hankcs.hanlp.HanLP;
import com.hankcs.hanlp.dictionary.py.Pinyin;

import analogy.Analogy;

public class SmTest {

	int dim = 100;
	Map<String, Integer> words = new HashMap<>(); // 输入的词向量
	ArrayList<ArrayList<Double>> C = new ArrayList<ArrayList<Double>>(); // 矩阵，一行一个词向量
	Map<String, Integer> twords = new HashMap<>(); // 词的辅助向量
	ArrayList<ArrayList<Double>> tC = new ArrayList<ArrayList<Double>>();
	static Map<String, Integer> pinyins = new HashMap<>(); // pinyin map
	static ArrayList<ArrayList<Double>> P = new ArrayList<ArrayList<Double>>();

	// String[] textPath = {"small", "large", "sogou", "wangyi"};
	String[] textPath = { "sogou", "wangyi" };
	String[] encode = { "GBK", "utf-8" };

	public static void main(String[] args) {
		// the first parameter is the word vector directory, the second
		// parameter is evaluate result file

		boolean isbach = false;
		boolean iswordsim = false;
		boolean isclassify = false;
		boolean isanalogy = true;
		boolean isfind = false;
		String trainFile = null;
		String resultFile = null;
		String word = null;
		int range = 5;
		String model = null;
		String dic = "pinyin";

		int j;
		if ((j = getParam("-iswordsim", args)) > 0) {//词相似性
			iswordsim = Boolean.valueOf(args[j]);
		}
		if ((j = getParam("-isclassify", args)) > 0) {//文本分类
			isclassify = Boolean.valueOf(args[j]);
		}
		if ((j = getParam("-isanalogy", args)) > 0) {//类比实验
			isanalogy = Boolean.valueOf(args[j]);
		}
		if ((j = getParam("-isfind", args)) > 0) {//查找近义词
			isfind = Boolean.valueOf(args[j]);
		}
		if ((j = getParam("-isbach", args)) > 0) {//处理同一模型的几份词向量文件
			isbach = Boolean.valueOf(args[j]);
		}
		if ((j = getParam("-input", args)) > 0) {//
			trainFile = args[j];
		}
		if ((j = getParam("-output", args)) > 0) {//
			resultFile = args[j];
		}
		if ((j = getParam("-range", args)) > 0) {//类比实验中，取概率前range个词作类比
			range = Integer.valueOf(args[j]);
		}
		if ((j = getParam("-word", args)) > 0) {//指定词word查找近义词的
			word = args[j];
		}
		if ((j = getParam("-model", args)) > 0) {//分类模型
			model = args[j];
		}
		if ((j = getParam("-dic", args)) > 0) {
			dic = args[j];
		}
		// if((j = getParam("-encode", args)) > 0) {
		// this.encode = args[j];
		// }
		System.out.println("isbach " + isbach);
		System.out.println("isclassify: " + isclassify);
		System.out.println("iswordsim: " + iswordsim);
		System.out.println("isanalogy: " + isanalogy);
		System.out.println("isfind: " + isfind);
		System.out.println("input: " + trainFile);
		System.out.println("output: " + resultFile);
		System.out.println("model: " + model);

		SmTest st = new SmTest();
		if (iswordsim) {
			st.wordSim(isbach, trainFile, resultFile);
		}

		if (isclassify) {
			st.textClassification(isbach, trainFile, resultFile, model);
		}

		if (isanalogy) {
			st.analogyEva(isbach, trainFile, resultFile, range);
		}

		if (isfind) {
			st.readVec(trainFile);
			double[] target = new double[100];
			int charsize = 1;
			int indexp = pinyins.get("xiang2");
			for (int i1 = 0; i1 < 100; i1++) {
				target[i1] = target[i1] + P.get(indexp).get(i1) / charsize;
			}
			// indexp = pinyins.get("suan4");
			// for (int i1 = 0; i1 < 100; i1++) {
			// target[i1] = target[i1] + P.get(indexp).get(i1) / charsize;
			// }
			// indexp = pinyins.get("ji1");
			// for (int i1 = 0; i1 < 100; i1++) {
			// target[i1] = target[i1] + P.get(indexp).get(i1) / charsize;
			// }
			st.findSimiliar("财富", dic);
		}
	}

	public void textClassification(boolean isbach, String trainFile, String resultFile, String model) {
		if (isbach) {
			try {
				for (int i = 0; i < textPath.length; i++) {
					File saveResult = new File(resultFile);
					BufferedWriter write = new BufferedWriter(new OutputStreamWriter(
							new FileOutputStream(saveResult + "/" + textPath[i] + model + "class.txt"), "utf-8"));
					write.write(
							"Evaluation accuracies (%) on text classification, type, precision, recall, f value and total precision\r\n");
					write.flush();

					File dir = new File(trainFile);
					if (dir.isDirectory()) {
						System.out.println("the parameter file is a directory");
					}
					File[] trainfiles = dir.listFiles();
					double s;

					for (File file : trainfiles) {
						write.write(file.getName() + " ");
						write.flush();

						TextClassification tc = new TextClassification(file.toString());
						s = tc.calculateAcc(textPath[i], encode[i], write);
						write.write(s + "\r\n");
						write.flush();
					}
				}
			} catch (UnsupportedEncodingException | FileNotFoundException e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			} catch (IOException e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			}
		} else {
			TextClassification tc = new TextClassification("sig(x+y)/serial/3pycbow.txt");
			// double s = tc.calculateAcc(textPath, encode);
			// System.out.println(s);
		}
	}

	public void analogyEva(boolean isbach, String trainFile, String resultFile, int range) {
		if (isbach) {
			try {
				File saveResult = new File(resultFile);
				BufferedWriter write = new BufferedWriter(
						new OutputStreamWriter(new FileOutputStream(saveResult), "utf-8"));
				write.write("Evaluation accuracies (%) on analogical reasoning\r\n");
				write.flush();

				File dir = new File(trainFile);
				if (dir.isDirectory()) {
					System.out.println("the parameter file is a directory");
				}
				File[] trainfiles = dir.listFiles();
				double s;

				for (File file : trainfiles) {
					Analogy analogy = new Analogy(file.toString(), range);
					s = analogy.accuracyAnalogy();
					write.write(file.getName() + " " + s + "\r\n");
					write.flush();
				}
			} catch (UnsupportedEncodingException | FileNotFoundException e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			} catch (IOException e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			}
		} else {
			// String trainfile = "/home/muhe/JWE/src/word_vec";
			String trainfile = "E:\\文本表示语料\\Chinese\\npycbow.txt";
			Analogy analogy = new Analogy(trainfile, range);
			System.out.println(analogy.accuracyAnalogy());
		}
	}

	public void wordSim(boolean isbach, String trainFile, String resultFile) {
		if (isbach) {
			try {
				File saveResult = new File(resultFile);
				BufferedWriter write = new BufferedWriter(
						new OutputStreamWriter(new FileOutputStream(saveResult), "utf-8"));
				write.write("left is 240, right is 297\r\n");
				write.flush();

				File dir = new File(trainFile);
				if (dir.isDirectory()) {
					System.out.println("the parameter file is a directory");
				}
				File[] trainfiles = dir.listFiles();
				String[] testDate = { "240.txt", "297.txt" };
				double s;

				for (File file : trainfiles) {
					for (int i = 0; i < 2; i++) {
						String evaluatefile = testDate[i];
						Spearman spe = new Spearman(file.toString(), evaluatefile);
						s = spe.computeSpearman();
						if (i == 0) {
							write.write(file.getName() + " " + s + " ");
						}
						if (i == 1) {
							write.write(s + "\r\n");
						}
						write.flush();
					}
				}
			} catch (UnsupportedEncodingException | FileNotFoundException e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			} catch (IOException e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			}
		} else {
			String[] testData = { "240.txt", "297.txt" };
			for (int i = 0; i < 2; i++) {
				String evaluatefile = testData[i];
				// String trainfile =
				// "/home/muhe/eclipse/workspace/word2MTR/sig(x+y)/serial/2pycbow.txt";
				String trainfile = "E:\\文本表示语料\\Chinese\\npycbow.txt";
				System.out.println(trainfile);
				Spearman spe = new Spearman(trainfile, evaluatefile);
				System.out.println(spe.computeSpearman());
			}
		}
	}

	public static int getParam(String para, String[] args) {
		int i;
		for (i = 0; i < args.length; i++) {
			if (args[i].equals(para)) {
				return i + 1;
			}
		}
		return -1;
	}

	void findSimiliar(double[] target, String dic) {
		// ./sig(x+y)/serial/our-model/2pycbow.txt
		double s;
		ArrayList<WeightedWords<String>> top5 = new ArrayList<WeightedWords<String>>();
		ArrayList<ArrayList<Double>> tempVec = null;
		int i = 0;
		double thetaw = 0.0, thetas;

		if (dic.equals("word")) {
			tempVec = tC;
		} else if (dic.equals("pinyin")) {
			tempVec = P;
		}

		if (dic.equals("word")) {
			for (int j = 0; j < dim; j++) {
				thetaw = thetaw + target[j] * target[j];
			}
		}

		for (Entry<String, Integer> e : twords.entrySet()) {
			s = 0.0;
			thetas = 0.0;
			int j = e.getValue();
			double[] temp = new double[dim];

			double charsizea = 0;
			List<Pinyin> pya = HanLP.convertToPinyinList(e.getKey());
			charsizea = e.getKey().length();
			for (Pinyin pinyin : pya) {
				int indexp = pinyins.get(pinyin.toString());
				for (int i1 = 0; i1 < dim; i1++) {
					temp[i1] = temp[i1] + P.get(indexp).get(i1) / charsizea;
				}
			}

			for (int j2 = 0; j2 < dim; j2++) {
				temp[j2] = temp[j2] + tempVec.get(j).get(j2);
			}

			for (int j2 = 0; j2 < dim; j2++) {
				s = s + target[j2] * temp[j2];
				thetas = thetas + temp[j2] * temp[j2];
			}
			s = s / (Math.sqrt(thetas) * Math.sqrt(thetaw));
			top5.add(new WeightedWords<String>(e.getKey(), s));
			Collections.sort(top5);
			if (top5.size() > 10) {
				top5.remove(10);
			}
		}

		for (int j = 0; j < top5.size(); j++) {
			System.out.println(top5.get(j).num + " " + top5.get(j).weight);
		}
	}

	void findSimiliar(String word, String dic) {
		// ./sig(x+y)/serial/our-model/2pycbow.txt
		System.out.println(word);
		double s;
		ArrayList<WeightedWords<String>> top5 = new ArrayList<WeightedWords<String>>();
		ArrayList<ArrayList<Double>> tempVec = null;
		int i = 0;
		double thetaw = 0.0, thetas;
		double[] target = new double[dim];

		if (dic.equals("word")) {
			tempVec = P;
			System.out.println(dic);
		} else if (dic.equals("pinyin")) {
			tempVec = P;
		}

		// if (word.matches("[\u4e00-\u9fa5]+")) {
		if (dic.equals("word")) {
			i = twords.get(word);
			for (int j = 0; j < dim; j++) {
				target[j] = tC.get(i).get(j);
			}

//			double charsizea = 0;
//			List<Pinyin> pya = HanLP.convertToPinyinList(word);
//			charsizea = word.length();
//
//			for (Pinyin pinyin : pya) {
//
//				int indexp = pinyins.get(pinyin.toString());
//				for (int j = 0; j < dim; j++) {
//					target[j] = target[j] + P.get(indexp).get(j) / charsizea;
//				}
//			}

			for (int j = 0; j < dim; j++) {
				thetaw = thetaw + target[j] * target[j];
			}

		} else {
			i = pinyins.get(word);
			for (int j = 0; j < dim; j++) {
				target[j] = tempVec.get(i).get(j);
				thetaw = thetaw + target[j] * target[j];
			}
		}

		for (Entry<String, Integer> e : pinyins.entrySet()) {
			s = 0.0;
			thetas = 0.0;
			int j = e.getValue();
			double[] temp = new double[dim];

			// double charsizea = 0;
			// List<Pinyin> pya = HanLP.convertToPinyinList(e.getKey());
			// charsizea = e.getKey().length();
			// for (Pinyin pinyin : pya) {
			// int indexp = pinyins.get(pinyin.toString());
			// for (int i1 = 0; i1 < dim; i1++) {
			// temp[i1] = temp[i1] + P.get(indexp).get(i1)/charsizea;
			// }
			// }

			for (int j2 = 0; j2 < dim; j2++) {
				temp[j2] = temp[j2] + tempVec.get(j).get(j2);
			}

			for (int j2 = 0; j2 < dim; j2++) {
				s = s + target[j2] * temp[j2];
				thetas = thetas + temp[j2] * temp[j2];
			}
			s = s / (Math.sqrt(thetas) * Math.sqrt(thetaw));
			top5.add(new WeightedWords<String>(e.getKey(), s));
			Collections.sort(top5);
			if (top5.size() > 10) {
				top5.remove(10);
			}
		}

		for (int j = 0; j < top5.size(); j++) {
			System.out.println(top5.get(j).num + " " + top5.get(j).weight);
		}
	}

	public void readVec(String trainfile) {
		try {
			InputStreamReader input = new InputStreamReader(new FileInputStream(trainfile), "utf-8");
			BufferedReader read = new BufferedReader(input);
			String line;
			String[] factors;
			int num = 0, pnum = 0, tnum = 0;
			line = read.readLine();
			factors = line.split(" ");
			// wa = Double.parseDouble(factors[0]);
			// wb = Double.parseDouble(factors[1]);
			while ((line = read.readLine()) != null) {
				if (line.equals("the word vector is ")) {
					continue;
				}
				if (line.equals("the pinyin vector is ") || line.equals("the temp vector is ")) {
					break;
				}
				factors = line.split(" ");
				words.put(factors[0], num);
				ArrayList<Double> vec = new ArrayList<Double>();
				for (int i = 1; i <= dim; i++) {
					vec.add(Double.valueOf(factors[i]));
				}
				C.add(vec);
				num++;
			}

			while ((line = read.readLine()) != null) {
				if (line.equals("the pinyin vector is ")) {
					break;
				}
				factors = line.split(" ");
				twords.put(factors[0], tnum);
				ArrayList<Double> vec = new ArrayList<Double>();
				for (int i = 1; i <= dim; i++) {
					vec.add(Double.valueOf(factors[i]));
				}
				tC.add(vec);
				tnum++;
			}

			while ((line = read.readLine()) != null) {
				factors = line.split(" ");
				pinyins.put(factors[0], pnum);
				ArrayList<Double> vec = new ArrayList<Double>();
				for (int i = 1; i <= dim; i++) {
					vec.add(Double.valueOf(factors[i]));
				}
				P.add(vec);
				pnum++;
			}

			System.out.println("C size " + C.size() + " " + num);
			System.out.println("tC size " + tC.size() + " " + tnum);
			System.out.println("P size " + P.size() + " " + pnum);
		} catch (UnsupportedEncodingException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		} catch (FileNotFoundException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	}
}
