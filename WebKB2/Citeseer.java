import java.io.File;
import java.io.FileWriter;
import java.util.HashMap;
import java.util.Map;
import java.util.Scanner;



public class Citeseer {
	public static void main(String ... args) throws Exception {
		Map<String, Integer> mapPapers = new HashMap<String, Integer>();
		Scanner sc = new Scanner(new File(args[0]));	//paperIds
		
		int i=1;
		while(sc.hasNextLine()) {
			String line = sc.nextLine();
			mapPapers.put(line, i);
			i++;
		}
		sc.close();
		System.out.println("Num of papers : " + mapPapers.size());
		
		sc = new Scanner(new File(args[1]));		//.cites file
		FileWriter writer = new FileWriter(new File(args[2]));		//output file
		while(sc.hasNextLine()) {
			String line = sc.nextLine();
			Scanner wordScanner = new Scanner(line);
			String paper1 = wordScanner.next();
			String paper2 = wordScanner.next();
			wordScanner.close();
			
			Integer p1 = mapPapers.get(paper1);
			Integer p2 = mapPapers.get(paper2);
			if (p1!=null && p2!=null)
				writer.write(p1 + "\t" + p2 + "\n");
			System.out.println(mapPapers.get(paper1) + "\t" + mapPapers.get(paper2));
		}
		
		sc.close(); 	 
		writer.close();
	}
}
