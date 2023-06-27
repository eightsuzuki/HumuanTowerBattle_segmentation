using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.SceneManagement;
using System.IO;

public class GameOver : MonoBehaviour
{
    public static bool isGameOver = false;
    // Start is called before the first frame update
    void Start()
    {/*
        string[] files = Directory.GetFiles(
              @"Assets/Resources", "*.png", SearchOption.AllDirectories
              );
        foreach (string file in files)
        {
            File.SetAttributes(file, FileAttributes.Normal);
            File.Delete(file);
        }*/
    }


    // Update is called once per frame
    void Update()
    {
        if (Input.GetMouseButtonDown (0)) {
        SceneManager.LoadScene ("Main");
      }
    }
}
