package moa.clusterers.newsClusterer;

import com.yahoo.labs.samoa.instances.Instances;

import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;

/**
 * Created by Xirui on 5/14/2017.
 */
public class MoaNewsUtil {


    public static Instances loadARFF(String fileName) {
        try {
            BufferedReader reader = new BufferedReader(new FileReader(fileName));
            Instances instances = new Instances(reader, null);
            return instances;
        } catch (FileNotFoundException e) {
            e.printStackTrace();
        }
        return null;
    }

}
