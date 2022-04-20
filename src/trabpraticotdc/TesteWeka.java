import java.text.DecimalFormat;
import java.util.Random;
import weka.core.Instances;
import weka.core.Instance;
import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.converters.ConverterUtils.DataSource;
import weka.classifiers.trees.J48;
import weka.classifiers.Evaluation;
//import weka.classifiers.lazy.IBk;
import weka.classifiers.bayes.NaiveBayes;

public class TesteWeka {
    private String caminhoDados;
    private Instances dados;

    public TesteWeka(String caminhoDados) {
        this.caminhoDados = caminhoDados;
    }

    public void leDados() throws Exception {
        DataSource fonte = new DataSource(caminhoDados);
        dados = fonte.getDataSet();
        if (dados.classIndex() == -1)
            dados.setClassIndex(dados.numAttributes() - 1);
    }

    public String imprimeDados() {
        String texto = "Dados do dataset:\n";
        for (int i = 0; i < dados.numInstances(); i++) {
            Instance atual = dados.instance(i);
            texto += ((i + 1) + ": " + atual + "\n");
        }
        return texto;
    } // até aqui esta rodando, não precisa alterar

    public void imprimeResultado(double[] resultado) {
        DecimalFormat df = new DecimalFormat("#.##");
        for (int i = 0; i < resultado.length; i++) {
            System.out.println("Classificação: " + df.format(resultado[i]));
        }
    }

    public DenseInstance criaInstancia(int valores) {
        // cria uma instância com os valores especificados

        DenseInstance inst = new DenseInstance(valores);
        Attribute bateria = dados.attribute(0);
        Attribute bluetooth = dados.attribute(1);
        Attribute velocidade_multiprocessador = dados.attribute(2);
        Attribute dual_sim = dados.attribute(3);
        Attribute megapixels_frontal = dados.attribute(4);
        Attribute quatro_g = dados.attribute(5);
        Attribute memoria_interna = dados.attribute(6);
        Attribute profundidade = dados.attribute(7);
        Attribute peso = dados.attribute(8);
        Attribute nucleos_proces = dados.attribute(9);
        Attribute megapixels_principal = dados.attribute(10);
        Attribute px_altura = dados.attribute(11);
        Attribute px_largura = dados.attribute(12);
        Attribute memoria_ram = dados.attribute(13);
        Attribute altura_tela = dados.attribute(14);
        Attribute largura_tela = dados.attribute(15);
        Attribute tempo_ligacao = dados.attribute(16);
        Attribute tres_g = dados.attribute(17);
        Attribute touch_screen = dados.attribute(18);
        Attribute wifi = dados.attribute(19);

        inst.setValue(bateria, 1520);
        inst.setValue(bluetooth, 1);
        inst.setValue(velocidade_multiprocessador, 2.2);
        inst.setValue(dual_sim, 0);
        inst.setValue(megapixels_frontal, 5);
        inst.setValue(quatro_g, 1);
        inst.setValue(memoria_interna, 33);
        inst.setValue(profundidade, 0.5);
        inst.setValue(peso, 177);
        inst.setValue(nucleos_proces, 8);
        inst.setValue(megapixels_principal, 18);
        inst.setValue(px_altura, 151);
        inst.setValue(px_largura, 1005);
        inst.setValue(memoria_ram, 3826);
        inst.setValue(altura_tela, 14);
        inst.setValue(largura_tela, 9);
        inst.setValue(tempo_ligacao, 13);
        inst.setValue(tres_g, 1);
        inst.setValue(touch_screen, 1);
        inst.setValue(wifi, 1);

        inst.setDataset(dados);
        return inst;
    }

    public String arvoreDeDecisaoJ48() throws Exception {

        // construindo o modelo
        J48 tree = new J48();
        tree.buildClassifier(dados);
        String retorno = "";
        retorno += tree;

        // avaliando sem validação cruzada o modelo construido
        retorno += "\nAvaliacao inicial: \n";
        Evaluation avaliacao;
        avaliacao = new Evaluation(dados);
        avaliacao.evaluateModel(tree, dados);
        retorno +=  "--> Instancias corretas: " + avaliacao.correct() + "\n";

        // avaliando com validação cruzada o modelo construido
        retorno +=  "Avaliacao cruzada: \n";
        Evaluation avalCruzada;
        avalCruzada = new Evaluation(dados);
        avalCruzada.crossValidateModel(tree, dados, 10, new Random(1));
        retorno += "--> Instancias corretas CV: " + avalCruzada.correct() + "\n";

        // Create empty instance with three attribute values
        Instance inst = criaInstancia(20);

        // Print the instance
        retorno += ("\nInstância: " + inst + "\n");

        double resultado = tree.classifyInstance(inst);
        retorno += ("Resultado: " + resultado);
        
        
        return retorno;
        /*
         * double resultado[] = tree.distributionForInstance(inst);
         * // System.out.println("Resultado: " + resultado[0]);
         * imprimeResultado(resultado);
         */
    }

    public String ClassificacaoNaiveBayes() throws Exception {
        // construindo o modelo
        NaiveBayes nb = new NaiveBayes();
        nb.buildClassifier(dados);
        String retorno = "";
        retorno += nb;

        // avaliando sem validação cruzada o modelo construido
       retorno += "\nAvaliacao inicial: \n";
        Evaluation avaliacao;
        avaliacao = new Evaluation(dados);
        avaliacao.evaluateModel(nb, dados);
        retorno += "--> Instancias corretas: " + avaliacao.correct() + "\n";

        /*
         * avaliacao=new Evaluation(dados);
         * avaliacao.evaluateModel(tree,dados);
         * System.out.println("-->Instancias corretas:"+avaliacao.correct()+"\n");
         * System.out.println("Avaliacao cruzada:\n");
         * this.evaInicial = avaliacao;
         * Evaluation avalCruzada;
         * avalCruzada=new Evaluation(dados);
         * avalCruzada.crossValidateModel(tree, dados, 10, new Random(1));
         * System.out.println("-->Instancias corretas CV:"+avalCruzada.correct()+"\n");
         * this.evaCruzada = avalCruzada;
         */
        // avaliando com validação cruzada o modelo construido
        retorno += "Avaliacao cruzada: \n";
        Evaluation avalCruzada;
        avalCruzada = new Evaluation(dados);
        avalCruzada.crossValidateModel(nb, dados, 10, new Random(1));
        retorno += "--> Instancias corretas CV: " + avalCruzada.correct() + "\n";

        // Create empty instance with three attribute values //Set instance's values for
        // the attributes "length", "weight", and "position"
        Instance inst = criaInstancia(20);

        // Print the instance
        retorno += "\nInstância: " + inst + "\n";

        double resultado = nb.classifyInstance(inst);
        retorno += "Resultado: " + resultado;
        
         return retorno;
        /*
         * double resultado[] = nb.distributionForInstance(inst);
         * 
         * imprimeResultado(resultado);
         */

    }

}
