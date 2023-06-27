using UnityEngine;

public class CameraController : MonoBehaviour
{

    public static bool isCollision;//衝突検出変数

    /// <summary>
    /// 衝突中を伝えるメソッド
    /// </summary>
    /// <param name="collision"></param>
    private void OnTriggerEnter2D(Collider2D collision)
    {
        Debug.Log("true_start");
        isCollision = true;
        Debug.Log("true_fin");
    }

    /// <summary>
    /// 衝突終了を伝えるメソッド
    /// </summary>
    /// <param name="collision"></param>
    private void OnTriggerExit2D(Collider2D collision)
    {
        isCollision = false;
    }
}