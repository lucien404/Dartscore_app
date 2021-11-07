import UIKit
import ImagePicker

class ViewController: UIViewController, ImagePickerDelegate, UITextFieldDelegate {

  lazy var button: UIButton = self.makeButton()

//  lazy var urlTF: UITextField = self.makeUrlTF()
  

  lazy var btnGetPoint: UIButton = self.makeSendBtn()
  
  lazy var imageChoose: UIImageView = self.makeUIImage()
  
  lazy var waitingLabel: UILabel = self.makeWatingLb()
  
  lazy var gameData :GameData = GameData.shared
  
  lazy var gameView :GameView? = nil
  
  
  func setGameView(_gameView :GameView) {
    gameView = _gameView
  }
  
  override func viewDidLoad() {
    super.viewDidLoad()

    view.backgroundColor = UIColor.white
    view.addSubview(button)
    button.translatesAutoresizingMaskIntoConstraints = false

    view.addConstraint(
      NSLayoutConstraint(item: button, attribute: .right,
                         relatedBy: .equal, toItem: view,
                         attribute: .right, multiplier: 1,
                         constant: 0))

    view.addConstraint(
      NSLayoutConstraint(item: button, attribute: .top,
                         relatedBy: .equal, toItem: view,
                         attribute: .top, multiplier: 1,
                         constant: 20))
    
    
//    view.addSubview(urlTF)
//    urlTF.translatesAutoresizingMaskIntoConstraints = false
//
//    view.addConstraint(
//      NSLayoutConstraint(item: urlTF, attribute: .bottom,
//                         relatedBy: .equal, toItem: view,
//                         attribute: .bottom, multiplier: 1,
//                         constant: 0))

    
    //getPonit Btn
    view.addSubview(btnGetPoint)
    btnGetPoint.translatesAutoresizingMaskIntoConstraints = false

    view.addConstraint(
      NSLayoutConstraint(item: btnGetPoint, attribute: .right,
                         relatedBy: .equal, toItem: view,
                         attribute: .right, multiplier: 1,
                         constant: 0))

    view.addConstraint(
      NSLayoutConstraint(item: btnGetPoint, attribute: .bottom,
                         relatedBy: .equal, toItem: view,
                         attribute: .bottom, multiplier: 1,
                         constant: -20))
    
    
    view.addSubview(imageChoose)
    view.sendSubviewToBack(imageChoose)
    view.addConstraint(
      NSLayoutConstraint(item: imageChoose, attribute: .centerX,
                         relatedBy: .equal, toItem: view,
                         attribute: .centerX, multiplier: 1,
                         constant: 0))
    view.addConstraint(
      NSLayoutConstraint(item: imageChoose, attribute: .centerY,
                         relatedBy: .equal, toItem: view,
                         attribute: .centerY, multiplier: 1,
                         constant: 0))
    
    view.addSubview(waitingLabel)
    waitingLabel.translatesAutoresizingMaskIntoConstraints = false
    view.addConstraint(
      NSLayoutConstraint(item: waitingLabel, attribute: .centerX,
                         relatedBy: .equal, toItem: view,
                         attribute: .centerX, multiplier: 1,
                         constant: 0))
    view.addConstraint(
      NSLayoutConstraint(item: waitingLabel, attribute: .centerY,
                         relatedBy: .equal, toItem: view,
                         attribute: .centerY, multiplier: 1,
                         constant: 0))
    
    waitingLabel.isHidden = true
  }

  func makeButton() -> UIButton {
    let button = UIButton()
    button.setTitle("Choose Image", for: .normal)
    button.setTitleColor(UIColor.black, for: .normal)
    button.addTarget(self, action: #selector(buttonTouched(button:)), for: .touchUpInside)
    button.backgroundColor = UIColor.red;

    return button
  }

  func makeWatingLb() -> UILabel {
    let waitingLb = UILabel()
    waitingLb.frame = CGRect(x: 0,y: 0, width: self.view.bounds.size.width,height: 20)
    waitingLb.textColor = UIColor.white
    waitingLb.text = "Waiting...."
    waitingLb.textAlignment = NSTextAlignment.center
    waitingLb.backgroundColor = UIColor.black
    waitingLb.alpha=1
    return waitingLb
  }
  
  @objc func buttonTouched(button: UIButton) {
    showImagePicker()
  }
  
  func showImagePicker() {
    let config = ImagePickerConfiguration()
    config.doneButtonTitle = "Choose"
    config.noImagesTitle = "Sorry! There are no images here!"
    config.recordLocation = false
    config.allowVideoSelection = true

    let imagePicker = ImagePickerController(configuration: config)
    imagePicker.delegate = self

    present(imagePicker, animated: true, completion: nil)
  }

  //
  func makeSendBtn() -> UIButton {
    let button = UIButton()
    button.setTitle("Get Points", for: .normal)
    button.setTitleColor(UIColor.black, for: .normal)
    button.addTarget(self, action: #selector(btnGetPoint(button:)), for: .touchUpInside)
    button.backgroundColor = UIColor.red;

    return button
  }
  
  @objc func btnGetPoint(button: UIButton) {
//    gameView!.TEST(_SCORE: 10000)
//      self.dismiss(animated: true, completion: nil)
//    gameData.addPlayScore(_score: 10)
//    gameView?.setPlayer()
//    self.dismiss(animated: true, completion: nil)
    if (imageChoose.image != nil)  {
      upload(img: imageChoose.image!)
    }
    else{
      showErrorTip()
    }
    
  }
  
  
  func makeUIImage() -> UIImageView {
    let screenh = UIScreen.main.applicationFrame.size.height
    let screenw = UIScreen.main.applicationFrame.size.width

    let vImage = UIImageView(frame: CGRect(x: 0, y: 0, width: screenw, height: screenh))
    vImage.tag = 100
    vImage.contentMode = UIView.ContentMode.scaleAspectFit
    return vImage
  }
  
  func upload(img:UIImage)
  {
    let data = img.pngData()
    let uploadurl:String="https://cv-feibiao.test.madnessglobalgame.com/feibiaocv"
    let request=NSMutableURLRequest(url:NSURL(string:uploadurl)! as URL)
    request.httpMethod="POST"
    
    let boundary:String="boundary"
    let contentType:String="multipart/form-data;boundary="+boundary
    request.addValue(contentType, forHTTPHeaderField:"Content-Type")
    let body=NSMutableData()
    let uuid = UUID().uuidString
    let strPlayerName = NSString(format:"Content-Disposition:form-data;name= player1" as NSString)
    body.append(strPlayerName.data(using: String.Encoding.utf8.rawValue)!)

    let strfile = NSString(format:"Content-Disposition:form-data;name=\"file\";filename=\"\(uuid).jpg\"\r\n" as NSString, uuid)
    print(strfile)
    body.append(NSString(format:"\r\n--\(boundary)\r\n" as NSString).data(using: String.Encoding.utf8.rawValue)!)
    
    body.append(strfile.data(using: String.Encoding.utf8.rawValue)!)

    body.append(NSString(format:"Content-Type:image/jpg\r\n\r\n").data(using: String.Encoding.utf8.rawValue)!)

    body.append(data!)

    body.append(NSString(format:"\r\n--\(boundary)--" as NSString).data(using: String.Encoding.utf8.rawValue)!)

    request.httpBody=body as Data

    let que=OperationQueue()
    waitingLabel.isHidden = false
    
    NSURLConnection.sendAsynchronousRequest(request as URLRequest, queue: que, completionHandler: { [self]

          (response, data, error) ->Void in
         if (error != nil)
         {
            DispatchQueue.main.async {
              let alertController = UIAlertController(title: "error",
                                                      message: nil, preferredStyle: .alert)
              //显示提示框
              self.present(alertController, animated: true, completion: nil)
              //两秒钟后自动消失
              DispatchQueue.main.asyncAfter(deadline: DispatchTime.now() + 2) {
                  self.presentedViewController?.dismiss(animated: false, completion: nil)
              }
            }
         }
         else
         {
          
            var dict: NSDictionary? = nil
            var score :String? = nil
           do{
            dict =  try JSONSerialization.jsonObject(with: data!, options: JSONSerialization.ReadingOptions.mutableLeaves) as? NSDictionary
            score = dict!["score"] as! String
            
            let array = getArrayFromJSONString(jsonString: score!)
  //          let adict1: NSDictionary  = dict![1] as! NSDictionary
            var totleScore :Int = 0
            for i in 0 ..< array.count {
              let count = array[i] as!Int
              totleScore = totleScore + count
            }
            gameData.addPlayScore(_score: totleScore)
              print(dict)
  //          var totleScore = 0
  //
  //          for item in score {
  //            totleScore = totleScore +item
  //          }
  //          gameData.addPlayScore(totleScore)
              DispatchQueue.main.async
              {
                let alertController = UIAlertController(title: score,
                                                        message: nil, preferredStyle: .alert)
                //显示提示框
                self.present(alertController, animated: true, completion: nil)
                //两秒钟后自动消失
                DispatchQueue.main.asyncAfter(deadline: DispatchTime.now() + 2) {
                    self.presentedViewController?.dismiss(animated: false, completion: nil)
                    gameView?.setPlayer()
  //                gameView!.TEST(_SCORE: 10000)
                    self.dismiss(animated: true, completion: nil)
                }
              }
           }catch{
            score = "error"
            showErrorTip()
           }
         }
          DispatchQueue.main.async {
            self.waitingLabel.isHidden = true
          }
      }
    )
  }
  
  func showErrorTip() {
    let alertController = PCLBlurEffectAlertController(title: "Oops!",
                                                       message: "The system can’t recognize the dart board, please change the angle or zoom in and shoot again (side view)", style: PCLBlurEffectAlert.ControllerStyle.alert)
               let image = UIImage(named: "tip.jpg")

    alertController.addImageView(with: image!)
            
                let catAction = PCLBlurEffectAlertAction(title: "Ok", style: .default) { _ in
                    print("You pressed Ok")
                }
                alertController.addAction(catAction)
                alertController.show()
  }
  
  func getArrayFromJSONString(jsonString:String) ->NSArray{
           
          let jsonData:Data = jsonString.data(using: .utf8)!
           
          let array = try? JSONSerialization.jsonObject(with: jsonData, options: .mutableContainers)
          if array != nil {
              return array as! NSArray
          }
          return array as! NSArray
           
      }

  // MARK: - ImagePickerDelegate

  func cancelButtonDidPress(_ imagePicker: ImagePickerController) {
    imagePicker.dismiss(animated: true, completion: nil)
  }

  func wrapperDidPress(_ imagePicker: ImagePickerController, images: [UIImage]) {
    /*
    guard images.count > 0 else { return }

    let lightboxImages = images.map {
      return LightboxImage(image: $0)
    }

    let lightbox = LightboxController(images: lightboxImages, startIndex: 0)
    imagePicker.present(lightbox, animated: true, completion: nil)
     */
  }

  func doneButtonDidPress(_ imagePicker: ImagePickerController, images: [UIImage]) {
    let nImageCount = images.count
    let imageUse = images[nImageCount - 1]

    imageChoose.image = imageUse
    imagePicker.dismiss(animated: true, completion: nil)
  }
}
