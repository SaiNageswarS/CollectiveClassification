import java.io.File;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Map;
import java.util.Scanner;
import java.util.Set;


public class CiteseerLinkGen {

	/**
	 * @param args
	 */
	public static void main(String[] args) throws Exception {
		// TODO Auto-generated method stub
		Scanner sc = new Scanner(new File(args[0]));
		Map<Integer, Set<Integer>> edges = new HashMap<Integer, Set<Integer>>();
		
		while(sc.hasNextLine()) {
			String line = sc.nextLine();
			Scanner wordScanner = new Scanner(line);
			int node2 = wordScanner.nextInt();
			int node1 = wordScanner.nextInt();
			wordScanner.close();
			
			Set<Integer> nodes = edges.get(node1);
			if(nodes==null) {
				nodes = new HashSet<Integer>();
				nodes.add(node1);
			}
			
			nodes.add(node2);
			edges.put(node1, nodes);
		}
		sc.close();

		//constructing sparse hypergraph incidence matrix
		//each entry in map is a edge
		int edgeNum=1;
		for(Integer n:edges.keySet()) {
			Set<Integer> nodes = edges.get(n);
			if (nodes.size()==1) 
				continue;
			for(int node:nodes) {
				Integer[] tuple = new Integer[3];
				tuple[0] = node;
				tuple[1] = edgeNum;
				tuple[2] = 1;
				System.out.println(tuple[0] + "\t" + tuple[1] + "\t" + tuple[2]);
			}
			edgeNum++;
		}
	}

}
